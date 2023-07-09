use std::arch::aarch64::*;
use std::mem::size_of;

const VEC_SIZE: usize = size_of::<uint8x16_t>();

/// Unroll size for mem{r}chr.
const UNROLL_SIZE_1: usize = 4;
/// Unroll size for mem{r}chr{2,3}.
const UNROLL_SIZE_23: usize = 2;

#[target_feature(enable = "neon")]
pub unsafe fn memchr(n1: u8, haystack: &[u8]) -> Option<usize> {
    memchr_generic_neon::<true, 1, { 2 * 1 }, { 4 * 1 }, UNROLL_SIZE_1>(
        [n1],
        haystack,
    )
}

#[target_feature(enable = "neon")]
pub unsafe fn memchr2(n1: u8, n2: u8, haystack: &[u8]) -> Option<usize> {
    memchr_generic_neon::<true, 2, { 2 * 2 }, { 4 * 2 }, UNROLL_SIZE_23>(
        [n1, n2],
        haystack,
    )
}

#[target_feature(enable = "neon")]
pub unsafe fn memchr3(
    n1: u8,
    n2: u8,
    n3: u8,
    haystack: &[u8],
) -> Option<usize> {
    memchr_generic_neon::<true, 3, { 2 * 3 }, { 4 * 3 }, UNROLL_SIZE_23>(
        [n1, n2, n3],
        haystack,
    )
}

#[target_feature(enable = "neon")]
pub unsafe fn memrchr(n1: u8, haystack: &[u8]) -> Option<usize> {
    memchr_generic_neon::<false, 1, { 2 * 1 }, { 4 * 1 }, UNROLL_SIZE_1>(
        [n1],
        haystack,
    )
}

#[target_feature(enable = "neon")]
pub unsafe fn memrchr2(n1: u8, n2: u8, haystack: &[u8]) -> Option<usize> {
    memchr_generic_neon::<false, 2, { 2 * 2 }, { 4 * 2 }, UNROLL_SIZE_23>(
        [n1, n2],
        haystack,
    )
}

#[target_feature(enable = "neon")]
pub unsafe fn memrchr3(
    n1: u8,
    n2: u8,
    n3: u8,
    haystack: &[u8],
) -> Option<usize> {
    memchr_generic_neon::<false, 3, { 2 * 3 }, { 4 * 3 }, UNROLL_SIZE_23>(
        [n1, n2, n3],
        haystack,
    )
}

const fn generate_mask64() -> u64 {
    let mut mask = 0;
    let mut byte = 0b0000_0001;

    let mut i = 0;
    while i < 8 {
        mask |= byte;
        byte <<= 8 + 1;

        i += 1;
    }

    mask
}

/// Returns true if the all bits in the register are set to 0.
#[inline(always)]
unsafe fn eq0(x: uint8x16_t) -> bool {
    low64(vpmaxq_u8(x, x)) == 0
}

#[inline(always)]
unsafe fn low64(x: uint8x16_t) -> u64 {
    vgetq_lane_u64(vreinterpretq_u64_u8(x), 0)
}

// .fold() and .reduce() cause LLVM to generate a huge dependency chain,
// so we need a custom function to explicitly parallelize the bitwise OR
// reduction to better take advantage of modern superscalar CPUs.
#[inline(always)]
unsafe fn parallel_reduce<const N: usize>(
    mut masks: [uint8x16_t; N],
) -> uint8x16_t {
    let mut len = masks.len();

    while len != 1 {
        for i in 0..len / 2 {
            masks[i] = vorrq_u8(masks[i * 2], masks[i * 2 + 1]);
        }
        if len & 1 != 0 {
            masks[0] = vorrq_u8(masks[0], masks[len - 1]);
        }
        len /= 2;
    }

    masks[0]
}

/// Search 64 bytes
#[inline(always)]
unsafe fn search64<
    const IS_FWD: bool,
    const N: usize,
    const N2: usize,
    const N4: usize,
>(
    n: [u8; N],
    ptr: *const u8,
    start_ptr: *const u8,
) -> Option<usize> {
    assert!(N4 == 4 * N);
    assert!(N2 == 2 * N);

    const MASK4: u64 = generate_mask64();

    let repmask4 = vreinterpretq_u8_u64(vdupq_n_u64(MASK4));

    let x1 = vld1q_u8(ptr);
    let x2 = vld1q_u8(ptr.add(1 * VEC_SIZE));
    let x3 = vld1q_u8(ptr.add(2 * VEC_SIZE));
    let x4 = vld1q_u8(ptr.add(3 * VEC_SIZE));

    let nv = n.map(|x| vdupq_n_u8(x));

    let masks1 = nv.map(|x| vceqq_u8(x1, x));
    let masks2 = nv.map(|x| vceqq_u8(x2, x));
    let masks3 = nv.map(|x| vceqq_u8(x3, x));
    let masks4 = nv.map(|x| vceqq_u8(x4, x));

    let cmpmask = parallel_reduce({
        let mut mask1234 = [vdupq_n_u8(0); N4];
        mask1234[..N].copy_from_slice(&masks1);
        mask1234[N..2 * N].copy_from_slice(&masks2);
        mask1234[2 * N..3 * N].copy_from_slice(&masks3);
        mask1234[3 * N..4 * N].copy_from_slice(&masks4);
        mask1234
    });

    if !eq0(cmpmask) {
        let cmp1 = parallel_reduce(masks1);
        let cmp2 = parallel_reduce(masks2);
        let cmp3 = parallel_reduce(masks3);
        let cmp4 = parallel_reduce(masks4);

        let cmp1 = vandq_u8(repmask4, cmp1);
        let cmp2 = vandq_u8(repmask4, cmp2);
        let cmp3 = vandq_u8(repmask4, cmp3);
        let cmp4 = vandq_u8(repmask4, cmp4);

        let reduce1 = vpaddq_u8(cmp1, cmp2);
        let reduce2 = vpaddq_u8(cmp3, cmp4);
        let reduce3 = vpaddq_u8(reduce1, reduce2);
        let reduce4 = vpaddq_u8(reduce3, reduce3);

        let low64: u64 = low64(reduce4);

        let offset = ptr as usize - start_ptr as usize;

        if IS_FWD {
            return Some(offset + low64.trailing_zeros() as usize);
        } else {
            return Some(
                offset + (4 * VEC_SIZE - 1) - (low64.leading_zeros() as usize),
            );
        }
    }

    None
}

/// Search 32 bytes
#[inline(always)]
unsafe fn search32<const IS_FWD: bool, const N: usize, const N2: usize>(
    n: [u8; N],
    ptr: *const u8,
    start_ptr: *const u8,
) -> Option<usize> {
    assert!(N2 == 2 * N);

    let x1 = vld1q_u8(ptr);
    let x2 = vld1q_u8(ptr.add(VEC_SIZE));

    let nv = n.map(|x| vdupq_n_u8(x));

    let masks1 = nv.map(|x| vceqq_u8(x1, x));
    let masks2 = nv.map(|x| vceqq_u8(x2, x));

    let cmpmask = parallel_reduce({
        let mut mask12 = [vdupq_n_u8(0); N2];
        mask12[..N].copy_from_slice(&masks1);
        mask12[N..2 * N].copy_from_slice(&masks2);
        mask12
    });

    if !eq0(cmpmask) {
        let cmp1 = parallel_reduce(masks1);
        let cmp2 = parallel_reduce(masks2);
        let combined1 = vshrn_n_u16(vreinterpretq_u16_u8(cmp1), 4);
        let combined2 = vshrn_n_u16(vreinterpretq_u16_u8(cmp2), 4);
        let bits1 = vget_lane_u64(vreinterpret_u64_u8(combined1), 0);
        let bits2 = vget_lane_u64(vreinterpret_u64_u8(combined2), 0);

        let offset = ptr as usize - start_ptr as usize;

        if IS_FWD {
            let byte_offset = if bits1 != 0 {
                (bits1.trailing_zeros() / 4) as usize
            } else {
                let bits2 = vget_lane_u64(vreinterpret_u64_u8(combined2), 0);
                VEC_SIZE + (bits2.trailing_zeros() / 4) as usize
            };
            return Some(offset + byte_offset);
        } else {
            let byte_offset = if bits2 != 0 {
                (bits2.leading_zeros() / 4) as usize
            } else {
                VEC_SIZE + (bits1.leading_zeros() / 4) as usize
            };
            return Some(offset + (2 * VEC_SIZE - 1) - byte_offset);
        }
    }

    None
}

/// Search 16 bytes
#[inline(always)]
unsafe fn search16<const IS_FWD: bool, const N: usize>(
    n: [u8; N],
    ptr: *const u8,
    start_ptr: *const u8,
) -> Option<usize> {
    let nv = n.map(|x| vdupq_n_u8(x));

    let x1 = vld1q_u8(ptr);

    let cmp_masks = nv.map(|x| vceqq_u8(x1, x));

    let cmpmask = parallel_reduce(cmp_masks);

    let combined = vshrn_n_u16(vreinterpretq_u16_u8(cmpmask), 4);
    let comb_low = vget_lane_u64(vreinterpret_u64_u8(combined), 0);
    if comb_low != 0 {
        let offset = ptr as usize - start_ptr as usize;

        let res = if IS_FWD {
            offset + comb_low.trailing_zeros() as usize / 4
        } else {
            offset + (VEC_SIZE - 1) - (comb_low.leading_zeros() as usize / 4)
        };
        return Some(res);
    }

    None
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn memchr_generic_neon<
    const IS_FWD: bool,
    const N: usize,
    const N2: usize,
    const N4: usize,
    const UNROLL: usize,
>(
    n: [u8; N],
    haystack: &[u8],
) -> Option<usize> {
    assert!(UNROLL <= 4 && UNROLL.is_power_of_two());

    let is_match = |x: u8| -> bool { n.iter().any(|&y| y == x) };

    let start_ptr = haystack.as_ptr();

    if haystack.len() < VEC_SIZE {
        if IS_FWD {
            // For whatever reason, LLVM generates significantly worse
            // code when using .copied() on the forward search, but
            // generates very good code for the reverse search (even
            // better than manual pointer arithmetic).
            return haystack.iter().position(|&x| is_match(x));
        } else {
            return haystack.iter().copied().rposition(is_match);
        }
    }

    // dynamic trait object devirtualized by LLVM upon monomorphization
    let iter: &mut dyn Iterator<Item = &[u8]>;

    let mut x1;
    let mut x2;
    let remainder;

    if IS_FWD {
        let temp = haystack.chunks_exact(UNROLL * VEC_SIZE);
        remainder = temp.remainder();
        x1 = temp;
        iter = &mut x1;
    } else {
        let temp = haystack.rchunks_exact(UNROLL * VEC_SIZE);
        remainder = temp.remainder();
        x2 = temp;
        iter = &mut x2;
    }

    let loop_search = match UNROLL {
        1 => search16::<IS_FWD, N>,
        2 => search32::<IS_FWD, N, N2>,
        4 => search64::<IS_FWD, N, N2, N4>,
        _ => unreachable!(),
    };

    for chunk in iter {
        if let Some(idx) = loop_search(n, chunk.as_ptr(), start_ptr) {
            return Some(idx);
        }
    }

    let mut ptr = if IS_FWD {
        remainder.as_ptr()
    } else {
        remainder.as_ptr().add(remainder.len()).offset(-(VEC_SIZE as isize))
    };

    if UNROLL > 1 {
        for _ in 0..remainder.len() / VEC_SIZE {
            if let Some(idx) = if IS_FWD {
                let ret = search16::<IS_FWD, N>(n, ptr, start_ptr);

                ptr = ptr.add(VEC_SIZE);

                ret
            } else {
                let ret = search16::<IS_FWD, N>(n, ptr, start_ptr);

                ptr = ptr.offset(-(VEC_SIZE as isize));

                ret
            } {
                return Some(idx);
            }
        }
    }

    if haystack.len() % VEC_SIZE != 0 {
        // overlapped search of remainder
        if IS_FWD {
            return search16::<IS_FWD, N>(
                n,
                start_ptr.add(haystack.len() - VEC_SIZE),
                start_ptr,
            );
        } else {
            return search16::<IS_FWD, N>(n, start_ptr, start_ptr);
        }
    }

    None
}
