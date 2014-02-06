// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unordered containers, implemented as hash-tables (`HashSet` and `HashMap` types)
//!
//! The tables use a keyed hash with new random keys generated for each container, so the ordering
//! of a set of keys in a hash table is randomized.
//!
//! # Example
//!
//! ```rust
//! use std::hashmap::HashMap;
//!
//! // type inference lets us omit an explicit type signature (which
//! // would be `HashMap<&str, &str>` in this example).
//! let mut book_reviews = HashMap::new();
//!
//! // review some books.
//! book_reviews.insert("Adventures of Hucklebury Fin",      "My favorite book.");
//! book_reviews.insert("Grimms' Fairy Tales",               "Masterpiece.");
//! book_reviews.insert("Pride and Prejudice",               "Very enjoyable.");
//! book_reviews.insert("The Adventures of Sherlock Holmes", "Eye lyked it alot.");
//!
//! // check for a specific one.
//! if !book_reviews.contains_key(& &"Les Misérables") {
//!     println!("We've got {} reviews, but Les Misérables ain't one.",
//!              book_reviews.len());
//! }
//!
//! // oops, this review has a lot of spelling mistakes, let's delete it.
//! book_reviews.remove(& &"The Adventures of Sherlock Holmes");
//!
//! // look up the values associated with some keys.
//! let to_find = ["Pride and Prejudice", "Alice's Adventure in Wonderland"];
//! for book in to_find.iter() {
//!     match book_reviews.find(book) {
//!         Some(review) => println!("{}: {}", *book, *review),
//!         None => println!("{} is unreviewed.", *book)
//!     }
//! }
//!
//! // iterate over everything.
//! for (book, review) in book_reviews.iter() {
//!     println!("{}: \"{}\"", *book, *review);
//! }
//! ```
use cmp::{Eq, Equiv};
use clone::Clone;
use container::*;
use default::Default;
use fmt;
use hash::Hash;
use iter;
use iter::*;
use num;
use option::*;
use prelude::{Drop, Map};
use ptr;
use ptr::{RawPtr};
use rand;
use rand::Rng;
use result::*;
use str::*;
use unstable::intrinsics::{transmute, forget};
use util;
use vec::{ImmutableVector, MutableVector};
use vec_ng;

// We use this type for the load factor, to avoid floating point operations
// which might not be supported efficiently on some hardware.
type Fraction = (u64, u64); // (numerator, denominator)

// multiplication by a fraction, in a way that won't generally overflow for
// array sizes outside a factor of 10 of U64_MAX.
#[inline]
fn fraction_mul(lhs: uint, rhs: Fraction) -> uint {
    let (num, den) = rhs;
    (((lhs as u64) * num) / den) as uint
}

static INITIAL_LOG2_CAP: uint = 5;
static INITIAL_CAPACITY: uint = 1 << INITIAL_LOG2_CAP; // 2^5
static EMPTY_BUCKET: u64 = 0u64;
static INITIAL_LOAD_FACTOR: Fraction = (9, 10);

// The main performance trick in this hashmap is called Robin Hood Hashing.
// It gains its excellent performance from one key invariant:
//
//    If an insertion collides with an existing element, and that elements
//    "probe distance" (how far away the element is from its ideal location)
//    is higher than how far we've already probed, swap the elements.
//
// This massively lowers variance in probe distance, and allows us to get very
// high load factors with good performance. The 90% load factor I use is rather
// conservative.
//
// > Why a load factor of 90%?
//
// In general, all the distances to inital buckets will converge on the mean.
// At a load factor of α, the odds of finding the target bucket after k
// probes is approximately 1-α^k. If we set this equal to 50% (since we converge
// on the mean) and set k=8 (64-byte cache line / 8-byte hash), α=0.92. I round
// this down to 0.90 to make the math easier on the CPU and avoid its FPU.
// Since on average we start the probing in the middle of a cache line, this
// strategy pulls in two cache lines of hashes on every lookup. I think that's
// pretty good, but if you want to trade off some space, it could go down to one
// cache line of average with an α of 0.84.
//
// > Wait, what? Where did you get 1-α^k from?
//
// On the first probe, your odds of a collision with an existing element is α.
// The odds of doing this twice in a row is approximatelly α^2. For three times,
// α^3, etc. Therefore, the odds of colliding k times is α^k. The odds of NOT
// colliding after k tries is 1-α^k.
//
// Relevant papers/articles:
//
// http://codecapsule.com/2013/11/11/robin-hood-hashing/
// http://codecapsule.com/2013/11/17/robin-hood-hashing-backward-shift-deletion/
// https://cs.uwaterloo.ca/research/tr/1986/CS-86-14.pdf <- original paper
//
// FIXME(cgaebel): An additional optimization that I have not yet implemented
// or explored is keeping track of the distance-to-bucket histogram as explored
// in the original paper. I'm suspicious of it because it uses a map internally.
// If this map were replaced with a hashmap, it could be faster, but now our
// data structure is self-referential and blows up. Also, this allows very good
// first guesses, but array accesses are no longer linear and in one direction
// in the general case.

/// A hash map implementation which uses linear probing with Robin Hood bucket
/// stealing. The hashes are all keyed by the task-local random number generator
/// on creation by default. This can be overriden with one of the constructors.
///
/// It is required that the keys implement the `Eq` and `Hash` traits, although
/// this can frequently be achieved by just implementing the `Eq` and
/// `IterBytes` traits as `Hash` is automatically implemented for types that
/// implement `IterBytes`.
pub struct HashMap<K, V> {
    // All hashes are keyed on these values, to prevent hash collision attacks.
    priv k0: u64,
    priv k1: u64,

    // When size == grow_at, we double the capacity.
    priv grow_at: uint,

    // When size == shrink_at, we halve the capacity.
    priv shrink_at: uint,

    // The capacity must never drop below this.
    priv minimum_capacity: uint,

    // To convert a hash to an index, we exploit the fact that all these
    // vectors must be a power of 2 in size. The hash mask is:
    //
    // log2(capacity) - 1
    //
    // then, k % capacity == k & hash_mask. This lets us skip idivs, which
    // are very expensive to do several times on every operation.
    priv hash_mask: u64,

    priv load_factor: Fraction,
    priv size: uint,

    // These vectors are unzipped to reduce cache pressure.

    // if hash == EMPTY_BUCKET, that spot is blank.
    priv hashes: vec_ng::Vec<u64>,

    // An entry at index 'i' is "undefined" if hashes[i] == EMPTY_BUCKET.
    priv keys:   vec_ng::Vec<K>,
    priv vals:   vec_ng::Vec<V>,
}

fn make_uninitialized_vec<T>(capacity: uint) -> vec_ng::Vec<T> {
    unsafe {
        let mut ret = vec_ng::Vec::with_capacity(capacity);
        ret.set_len(capacity);
        ret
    }
}

/// Get the number of elements which will force the capacity to grow.
#[inline]
fn grow_at(capacity: uint, load_factor: Fraction) -> uint {
    fraction_mul(capacity, load_factor)
}

// FIXME(cgaebel): Make this user configurable.
/// Get the number of elements which will force the capacity to shrink.
#[inline]
fn shrink_at(capacity: uint) -> uint {
    capacity >> 2
}

/// Builds a hash_mask for any power of two. This is extremely fast
// (just a subtraction by 1), but less general than build_hashmask_from_uint.
fn build_hashmask_from_pow2(capacity: uint) -> u64 {
    let res = capacity - 1;
    return res as u64
}

/// Build a hash_mask from an arbitrary uint. This is more
/// expensive than build_hashmask_from_pow2, but more general.
fn build_hashmask_from_uint(value: uint) -> u64 {
    build_hashmask_from_pow2(num::next_power_of_two(value))
}

#[test]
fn sane_hashmask() {
    assert_eq!(build_hashmask_from_uint(14), 0xFu64);
    assert_eq!(build_hashmask_from_pow2(16), 0xFu64);
    assert_eq!(build_hashmask_from_uint(16), 0xFu64);
    assert_eq!(build_hashmask_from_uint(1 << 5), 31u64);
}

/// Turns a hash into an index into the underlying arrays.
/// This exploits the power-of-two size of the hashtable with the
/// hash_mask member to prevent a costly division.
/// Fun fact: IDIV is ~30 cycles on Sandy Bridge. That's a 15 ns pipeline
///           stall!
#[inline]
fn hash_to_index(hash: u64, hash_mask: u64) -> uint {
    (hash & hash_mask) as uint
}

/// Move the probe to the next slot in the hashtable
#[inline]
fn probe_next(probe: uint, hash_mask: u64) -> uint {
    probe_skip(probe, hash_mask, 1)
}

/// Move the probe forward 'skip' spaces.
#[inline]
fn probe_skip(probe: uint, hash_mask: u64, skip: uint) -> uint {
    (probe + skip) & (hash_mask as uint)
}

#[inline]
fn get_vec_ref<'a, T>(v: &'a vec_ng::Vec<T>, idx: uint) -> &'a T {
    unsafe { v.as_slice().unsafe_ref(idx) }
}

#[inline]
fn get_vec_mut_ref<'a, T>(v: &'a mut vec_ng::Vec<T>, idx: uint) -> &'a mut T {
    unsafe { v.as_mut_slice().unsafe_mut_ref(idx) }
}

#[inline]
fn take_vec_elem<T>(v: &mut vec_ng::Vec<T>, idx: uint) -> T {
    unsafe {
        ptr::read_and_zero_ptr(v.as_mut_slice().as_mut_ptr().offset(idx as int))
    }
}

#[inline]
fn put_vec_elem<T>(v: &mut vec_ng::Vec<T>, idx: uint, t: T) {
    unsafe {
        // Stolen from libstd/sync/deque.rs. I hope it works!
        let ptr = v.as_mut_slice().as_mut_ptr().offset(idx as int);
        ptr::copy_nonoverlapping_memory(ptr as *mut T, &t as *T, 1);
        forget(t);
    }
}

impl<K: Hash + Eq, V> HashMap<K, V> {
    /// Gets the length of the underlying arrays.
    #[inline]
    fn capacity(&self) -> uint {
        self.hashes.len()
    }

    // Huge performance/safety tradeoff in the unsafe blocks below.
    // We essentially mark uninitialized buckets in the hashtable
    // as "uninitialized" by rust and do no bounds checking on array
    // accesses. This gives us blazing fast performance, but we depend
    // HEAVILY on test coverage to ensure correctness of future changes.
    //
    // The original module's core functions (find, swap, pop) original
    // implementations were built and tested using vectors of options
    // for safety. This performance hacking was added on afterwards.
    //
    // tl;dr: Write tests for your changes!!!

    #[inline]
    fn get_key_ref<'a>(&'a self, idx: uint) -> &'a K {
        get_vec_ref(&self.keys, idx)
    }

    #[inline]
    fn get_key_mut_ref<'a>(&'a mut self, idx: uint) -> &'a mut K {
        get_vec_mut_ref(&mut self.keys, idx)
    }

    #[inline]
    fn get_val_ref<'a>(&'a self, idx: uint) -> &'a V {
        get_vec_ref(&self.vals, idx)
    }

    #[inline]
    fn get_val_mut_ref<'a>(&'a mut self, idx: uint) -> &'a mut V {
        get_vec_mut_ref(&mut self.vals, idx)
    }

    #[inline]
    fn take_key(&mut self, idx: uint) -> K {
        take_vec_elem(&mut self.keys, idx)
    }

    #[inline]
    fn take_val(&mut self, idx: uint) -> V {
        take_vec_elem(&mut self.vals, idx)
    }

    #[inline]
    fn put_key(&mut self, idx: uint, k: K) {
        put_vec_elem(&mut self.keys, idx, k)
    }

    #[inline]
    fn put_val(&mut self, idx: uint, v: V) {
        put_vec_elem(&mut self.vals, idx, v)
    }

    #[inline]
    fn get_hash<'a>(&'a self, idx: uint) -> u64 {
        *get_vec_ref(&self.hashes, idx)
    }

    #[inline]
    fn del_hash(&mut self, idx: uint) {
        self.set_hash(idx, EMPTY_BUCKET);
    }

    #[inline]
    fn set_hash(&mut self, idx: uint, h: u64) {
        *self.get_hash_mut_ref(idx) = h;
    }

    #[inline]
    fn get_hash_mut_ref<'a>(&'a mut self, idx: uint) -> &'a mut u64 {
        get_vec_mut_ref(&mut self.hashes, idx)
    }

    /// We need to remove hashes of 0. That's reserved for empty buckets.
    /// This function wraps up hash_keyed to prevent hashes of 0.
    #[inline]
    fn make_hash<T: Hash>(&self, t: &T) -> u64 {
        match t.hash_keyed(self.k0, self.k1) {
            // This constant is exceedingly likely to hash to the same
            // bucket, but it won't be counted as empty!
            EMPTY_BUCKET => 0x8000_0000_0000_0000,
            h            => h
        }
    }

    /// Get the distance of the bucket at the given index that it lies
    /// from its 'ideal' location.
    ///
    /// In the cited blog posts above, this is called the "distance to
    /// inital bucket", or DIB.
    #[inline]
    fn bucket_distance(&self, index_of_elem: uint) -> uint {
        let first_probe_index =
            hash_to_index(self.get_hash(index_of_elem), self.hash_mask);
        if first_probe_index <= index_of_elem {
             // probe just went forward
            index_of_elem - first_probe_index
        } else {
            // probe wrapped around the hashtable
            index_of_elem + (self.capacity() - first_probe_index)
        }
    }

    /// Search for a pre-hashed key.
    #[inline]
    fn search_hashed_generic(&self, hash: u64, is_match: |&K| -> bool) -> Option<uint> {
        let mut probe = hash_to_index(hash, self.hash_mask);

        for num_probes in range(0u, self.size) {
            let bucket_hash = self.get_hash(probe);

            // hit an empty bucket.
            if bucket_hash == EMPTY_BUCKET { return None }

            // We can finish the search early if we hit any bucket
            // with a lower distance to initial bucket than we've probed.
            if self.bucket_distance(probe) < num_probes {
                return None
            }

            // Found it!
            if bucket_hash == hash && is_match(self.get_key_ref(probe)) {
                return Some(probe)
            }

            probe = probe_next(probe, self.hash_mask);
        }

        return None
    }

    fn search_hashed(&self, hash: u64, k: &K) -> Option<uint> {
        self.search_hashed_generic(hash, |k_| *k == *k_)
    }

    fn search_equiv<Q: Hash + Equiv<K>>(&self, q: &Q) -> Option<uint> {
        self.search_hashed_generic(self.make_hash(q), |k| q.equiv(k))
    }

    /// Search for a key, yielding the index if it's found in the hashtable.
    /// If you already have the hash for the key lying around, use
    /// search_hashed.
    #[inline]
    fn search(&self, k: &K) -> Option<uint> {
        self.search_hashed(self.make_hash(k), k)
    }
}

impl<K: Hash + Eq, V> Container for HashMap<K, V> {
    /// Return the number of elements in the map
    #[inline]
    fn len(&self) -> uint { self.size }
}

impl<K: Hash + Eq, V> Mutable for HashMap<K, V> {
    /// Clear the map, removing all key-value pairs.
    fn clear(&mut self) {
        for i in range(0, self.capacity()) {
            if self.get_hash(i) == EMPTY_BUCKET { continue }

            self.del_hash(i);
            self.take_key(i);
            self.take_val(i);
        }

        self.minimum_capacity = self.size;
        self.size             = 0u;
    }
}

impl <K: Hash + Eq, V> Map<K, V> for HashMap<K, V> {
    #[inline]
    fn find<'a>(&'a self, k: &K) -> Option<&'a V> {
        self.search(k).map(|idx| self.get_val_ref(idx))
    }
}

impl <K: Hash + Eq, V> MutableMap<K, V> for HashMap<K, V> {
    #[inline]
    fn find_mut<'a>(&'a mut self, k: &K) -> Option<&'a mut V>{
        match self.search(k) {
            None      => None,
            Some(idx) => Some(self.get_val_mut_ref(idx))
        }
    }

    #[inline]
    fn swap(&mut self, k: K, v: V) -> Option<V> {
        let hash = self.make_hash(&k);
        self.swap_hashed(hash, k, v)
    }

    fn pop(&mut self, k: &K) -> Option<V> {
        if self.size == 0 {
            return None
        }

        self.make_some_room(self.size - 1);

        let starting_index =
            match self.search(k) {
                Some(idx) => idx,
                None      => {
                    return None
                }
            };

        let mut probe = starting_index;

        for _ in range(0u, self.size + 1) {
            let bucket_hash = self.get_hash(probe);

            if bucket_hash == EMPTY_BUCKET // empty bucket
                // bucket that isn't us, which has a probe distance of 0.
                || (probe != starting_index && self.bucket_distance(probe) == 0)
            {
                // found the last bucket to shift.
                let ending_index = probe;

                self.del_hash(starting_index);
                self.take_key(starting_index);
                let retval = Some(self.take_val(starting_index));

                probe = starting_index;
                let mut next_probe = probe_next(probe, self.hash_mask);

                // backwards shift all the elements after the deleted
                // one.
                while next_probe != ending_index {
                    let next_hash = self.get_hash(next_probe);

                    self.set_hash(probe, next_hash);

                    // There could be nothing to shift in!
                    if next_hash == EMPTY_BUCKET {
                        self.take_key(probe);
                        self.take_val(probe);
                    } else {
                        self.del_hash(next_probe);
                        let old_key = self.take_key(next_probe);
                        self.put_key(probe, old_key);
                        let old_val = self.take_val(next_probe);
                        self.put_val(probe, old_val);
                    }

                    probe = next_probe;
                    next_probe = probe_next(probe, self.hash_mask);
                }

                // Empty out that last bucket.
                if self.get_hash(probe) != EMPTY_BUCKET {
                    self.del_hash(probe);
                    self.take_key(probe);
                    self.take_val(probe);
                }

                self.size -= 1;

                return retval;
            }

            probe = probe_next(probe, self.hash_mask);
        }

        // should really never happen.
        return None;
    }
}

impl<K: Hash + Eq, V> HashMap<K, V> {
    /// Create an empty HashMap
    pub fn new() -> HashMap<K, V> {
        HashMap::with_capacity(INITIAL_CAPACITY)
    }

    /// Create an empty HashMap with space for at least `capacity`
    /// elements in the hash table.
    // FIXME(cgaebel): Doesn't currently do anything useful.
    pub fn with_capacity(capacity: uint) -> HashMap<K, V> {
        let mut r = rand::task_rng();
        HashMap::with_capacity_and_keys(r.gen(), r.gen(), capacity)
    }

    /// Create an empty HashMap with space for at least `capacity`
    /// elements, using `k0` and `k1` as the keys.
    ///
    /// Warning: `k0` and `k1` are normally randomly generated, and
    /// are designed to allow HashMaps to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting them
    /// manually using this function can expose a DoS attack vector.
    pub fn with_capacity_and_keys(k0: u64, k1: u64, capacity: uint) -> HashMap<K, V> {
        let cap = num::max(INITIAL_CAPACITY, capacity);

        HashMap {
            k0:               k0,
            k1:               k1,
            load_factor:      INITIAL_LOAD_FACTOR,
            hash_mask:        build_hashmask_from_uint(cap),
            grow_at:          grow_at(cap, INITIAL_LOAD_FACTOR),
            shrink_at:        shrink_at(cap),
            minimum_capacity: cap,
            size:             0u,
            hashes:           vec_ng::Vec::from_elem(cap, EMPTY_BUCKET),
            keys:             make_uninitialized_vec(cap),
            vals:             make_uninitialized_vec(cap),
        }
    }

    /// The hashtable will never try to shrink below this size. You can use
    /// this function to reduce reallocations if your hashtable frequently
    /// grows and shrinks by large amounts.
    ///
    /// This function has no effect on the operational semantics of the
    /// hashtable, only on performance.
    #[inline]
    pub fn reserve_at_least(&mut self, new_minimum_capacity: uint) {
        let cap = num::max(INITIAL_CAPACITY, new_minimum_capacity);

        if self.capacity() < cap {
            self.resize(num::next_power_of_two(cap));
        }

        self.minimum_capacity = cap;
    }

    // FIXME(cgaebel): Allow the load factor to be changed dynamically,
    //                 and/or at initialization. I'm having trouble
    //                 figuring out a sane API for this without exporting
    //                 my hackish Fraction type, while avoiding floating
    //                 point.

    // FIXME(cgaebel): Would it be possible to reuse storage when we grow?
    //                 Since things would move if going into the "new" part
    //                 of the array when rehashed, this could possibly be
    //                 safely exploited.
    //
    //                 It would certainly make resize() more complicated.

    /// Resizes the internal vectors to a new capacity. It's your responsibility to:
    ///   1) Make sure the new capacity is enough for all the elements, accounting
    ///      for the load factor.
    ///   2) Ensure new_capacity is a power of two.
    fn resize(&mut self, new_capacity: uint) {
        let old_size = self.size;

        self.hash_mask   = build_hashmask_from_pow2(new_capacity);
        self.grow_at     = grow_at(new_capacity, self.load_factor);
        self.shrink_at   = shrink_at(new_capacity);
        self.size        = 0u;

        let cap = self.capacity();

        let mut old_hashes = util::replace(&mut self.hashes,
                                           vec_ng::Vec::from_elem(new_capacity, EMPTY_BUCKET));
        let mut old_keys = util::replace(&mut self.keys,
                                         make_uninitialized_vec(new_capacity));
        let mut old_vals = util::replace(&mut self.vals,
                                         make_uninitialized_vec(new_capacity));

        let mut num_copied = 0u;

        for i in range(0, cap) {
            let h = *get_vec_ref(&old_hashes, i);

            if h == EMPTY_BUCKET { continue }

            let k = take_vec_elem(&mut old_keys, i);
            let v = take_vec_elem(&mut old_vals, i);

            self.manual_insert_hashed_nocheck(h, k, v);

            num_copied += 1;
        }

        assert_eq!(old_size, num_copied);

        // Don't let the destructor run on any of the 'undefined' elements left
        // in the vector.
        unsafe {
            old_hashes.set_len(0);
            old_keys.set_len(0);
            old_vals.set_len(0);
        }
    }

    /// Performs any necessary resize operations, such that there's space for
    /// new_size elements.
    #[inline]
    fn make_some_room(&mut self, new_size: uint) {
        let should_shrink = new_size <= self.shrink_at;
        let should_grow   = self.grow_at <= new_size;

        if should_grow {
            let new_capacity = self.capacity() << 1;
            self.resize(new_capacity);
        } else if should_shrink {
            let new_capacity = self.capacity() >> 1;

            // Never shrink below the minimum capacity
            if self.minimum_capacity <= new_capacity {
                self.resize(new_capacity);
            }
        }
    }

    /// Fills a bucket. It must have been previously empty!
    #[inline]
    fn put_at(&mut self, probe: uint, hash: u64, k: K, v: V) {
        assert_eq!(self.get_hash(probe), EMPTY_BUCKET);

        self.set_hash(probe, hash);
        self.put_key(probe, k);
        self.put_val(probe, v);
        self.size += 1;
    }

    /// Perform robin hood bucket stealing at the given 'probe'. You must
    /// also pass that probe's "distance to initial bucket" so we don't have
    /// to recalculate it, as well as the total number of probes already done
    /// so we have some sort of upper bound on the number of probes to do.
    ///
    /// 'hash', 'k', and 'v' are the elements to robin hood into the hashtable.
    #[inline]
    fn robin_hood(&mut self, probe: uint, probe_dib: uint, num_probes: uint,
                  hash: u64, k: K, v: V) {
        let old_hash = self.get_hash(probe);

        // we must robin hood existing elements in the hashtable.
        assert!(old_hash != EMPTY_BUCKET);
        assert!(hash     != EMPTY_BUCKET);

        self.set_hash(probe, hash);

        let old_key = util::replace(self.get_key_mut_ref(probe), k);
        let old_val = util::replace(self.get_val_mut_ref(probe), v);

        self.insert_hashed_from(
            old_hash, old_key, old_val,
            probe_next(probe, self.hash_mask),
            probe_dib + 1, num_probes);
    }

    /// Manually insert a pre-hashed key-value pair, without first checking
    /// that there's enough room in the buckets. Returns a reference to the
    /// newly insert value.
    fn manual_insert_hashed_nocheck<'a>(
        &'a mut self, hash: u64, k: K, v: V) -> &'a mut V {

        let mut probe = hash_to_index(hash, self.hash_mask);

        let mut dib = 0u;

        for num_probes in range_inclusive(0u, self.size) {
            let bucket_hash = self.get_hash(probe);

            if bucket_hash == EMPTY_BUCKET {
                // Found a hole!
                self.put_at(probe, hash, k, v);
                return self.get_val_mut_ref(probe);
            }

            if bucket_hash == hash && k == *self.get_key_ref(probe) {
                // Key already exists. Get its reference.
                return self.get_val_mut_ref(probe);
            }

            let probe_dib = self.bucket_distance(probe);

            if probe_dib < dib {
                // Found a luckier bucket than me! Better steal his spot.
                self.robin_hood(
                    probe, probe_dib, num_probes,
                    hash, k, v);

                return self.get_val_mut_ref(probe);
            }

            dib += 1;
            probe = probe_next(probe, self.hash_mask);
        }

        // We really shouldn't be here.
        fail!("Internal HashMap error: Out of space.");
    }

    #[inline]
    fn manual_insert_hashed<'a>(&'a mut self, hash: u64, k: K, v: V) -> &'a mut V {
        self.make_some_room(self.size + 1);
        self.manual_insert_hashed_nocheck(hash, k, v)
    }

    /// Inserts an element, returning a reference to that element inside the
    /// hashtable.
    #[inline]
    fn manual_insert<'a>(&'a mut self, k: K, v: V) -> &'a mut V {
        let hash = self.make_hash(&k);
        self.manual_insert_hashed(hash, k, v)
    }

    /// Does a pre-hashed swap(), but without checking if there's enough
    /// room in the hashtable first. This will not resize the array.
    fn swap_hashed(&mut self, hash: u64, k: K, v: V) -> Option<V> {
        self.make_some_room(self.size + 1);

        let mut probe = hash_to_index(hash, self.hash_mask);

        let mut dib = 0u;

        for num_probes in range_inclusive(0u, self.size) {
            let bucket_hash = self.get_hash(probe);

            if bucket_hash == EMPTY_BUCKET {
                // Found a hole!
                self.put_at(probe, hash, k, v);
                return None;
            }

            if bucket_hash == hash && k == *self.get_key_ref(probe) {
                // Key already exists. Replace it.
                return Some(util::replace(self.get_val_mut_ref(probe), v));
            }

            let probe_dib = self.bucket_distance(probe);

            if probe_dib < dib {
                // Found a luckier bucket than me! Better steal his spot.
                self.robin_hood(probe, probe_dib, num_probes, hash, k, v);
                return None;
            }

            dib += 1;
            probe = probe_next(probe, self.hash_mask);
        }

        // We really shouldn't be here.
        fail!("Internal HashMap error: Out of space.");
    }

    // MUST NOT be used on elements that already exist in the table.
    // Also assumes that there's enough room in the table to do the
    // insertion.
    // This function is responsible for the robin hood bucket stealing.
    //
    // FIXME: The way this function is structured is kind of ugly.
    fn insert_hashed_from(
        &mut self,
        hash: u64,
        k: K,
        v: V,
        start_bucket: uint,
        dib_param: uint,
        iters_so_far: uint) {

        let mut probe = start_bucket;
        let mut dib   = dib_param;

        for num_probes in range_inclusive(iters_so_far, self.size) {
            let bucket_hash = self.get_hash(probe);

            if bucket_hash == EMPTY_BUCKET {
                // Finally. A hole!
                self.put_at(probe, hash, k, v);
                return;
            }

            let probe_dib = self.bucket_distance(probe);

            if probe_dib < dib {
                // Robin hood. Steal the spot.
                let old_hash = self.get_hash(probe);
                self.set_hash(probe, hash);

                let old_key  = util::replace(self.get_key_mut_ref(probe), k);
                let old_val  = util::replace(self.get_val_mut_ref(probe), v);

                // This had better be tail-call.
                return self.insert_hashed_from(
                    old_hash, old_key, old_val,
                    probe_next(probe, self.hash_mask),
                    probe_dib + 1, num_probes + 1);
            }

            dib += 1;
            probe = probe_next(probe, self.hash_mask);
        }
    }

    /// Return the value corresponding to the key in the map, or insert
    /// and return the value if it doesn't exist.
    #[inline]
    pub fn find_or_insert<'a>(&'a mut self, k: K, v: V) -> &'a mut V {
        match self.search(&k) {
            Some(idx) => self.get_val_mut_ref(idx),
            None      => self.manual_insert(k, v)
        }
    }

    /// Return the value corresponding to the key in the map, or create,
    /// insert, and return a new value if it doesn't exist.
    #[inline]
    pub fn find_or_insert_with<'a>(&'a mut self, k: K, f: |&K| -> V)
                               -> &'a mut V {
        match self.search(&k) {
            Some(idx) => self.get_val_mut_ref(idx),
            None      => {
                let v = f(&k);
                self.manual_insert(k, v)
            }
        }
    }

    /// Insert a key-value pair into the map if the key is not already present.
    /// Otherwise, modify the existing value for the key.
    /// Returns the new or modified value for the key.
    pub fn insert_or_update_with<'a>(
                                 &'a mut self,
                                 k: K,
                                 v: V,
                                 f: |&K, &mut V|)
                                 -> &'a mut V {
        match self.search(&k) {
            None      => self.manual_insert(k, v),
            Some(idx) => {
                let v = self.get_val_mut_ref(idx);
                f(&k, v);
                v
            }
        }
    }

    /// Retrieves a value for the given key, failing if the key is not
    /// present.
    #[inline]
    pub fn expect<'a>(&'a self, k: &K) -> &'a V {
        match self.find(k) {
            Some(v)   => v,
            None      => fail!("No entry found for key {:?}", k)
        }
    }

    /// Retrieves a value for the given key, failing if the key is not
    /// present.
    #[inline]
    pub fn expect_mut<'a>(&'a mut self, k: &K) -> &'a mut V {
        match self.find_mut(k) {
            Some(v) => v,
            None    => fail!("No entry found for key: {:?}", k)
        }
    }

    /// Deprecated. Use 'expect'.
    pub fn get<'a>(&'a self, k: &K) -> &'a V {
        self.expect(k)
    }

    /// Deprecated. Use 'expect_mut'.
    pub fn get_mut<'a>(&'a mut self, k: &K) -> &'a mut V {
        self.expect_mut(k)
    }

    /// Return true if the map contains a value for the specified key,
    /// using equivalence.
    #[inline]
    pub fn contains_key_equiv<Q:Hash + Equiv<K>>(&self, key: &Q) -> bool {
        self.find_equiv(key).is_some()
    }

    /// Return the value corresponding to the key in the map, using
    /// equivalence.
    #[inline]
    pub fn find_equiv<'a, Q:Hash + Equiv<K>>(&'a self, k: &Q)
                                             -> Option<&'a V> {
        match self.search_equiv(k) {
            None      => None,
            Some(idx) => Some(self.get_val_ref(idx))
        }
    }

    /// An iterator visiting all keys in arbitrary order.
    /// Iterator element type is &'a K.
    #[inline]
    pub fn keys<'a>(&'a self) -> Keys<'a, K, V> {
        self.iter().map(|(k, _v)| k)
    }

    /// An iterator visiting all values in arbitrary order.
    /// Iterator element type is &'a V.
    #[inline]
    pub fn values<'a>(&'a self) -> Values<'a, K, V> {
        self.iter().map(|(_k, v)| v)
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    /// Iterator element type is (&'a K, &'a V).
    #[inline]
    pub fn iter<'a>(&'a self) -> Entries<'a, K, V> {
        Entries { hashes:    &self.hashes
                , keys:      &self.keys
                , vals:      &self.vals
                , idx:       0
                , size_hint: self.size
                }
    }

    /// An iterator visiting all key-value pairs in arbitrary order,
    /// with mutable references to the values.
    /// Iterator element type is (&'a K, &'a mut V).
    #[inline]
    pub fn mut_iter<'a>(&'a mut self) -> MutEntries<'a, K, V> {
        MutEntries { hashes:    &self.hashes
                   , keys:      &self.keys
                   , vals:      &mut self.vals
                   , idx:       0
                   , size_hint: self.size
                   }
    }

    /// Creates a consuming iterator, that is, one that moves each key-value
    /// pair out of the map in arbitrary order. The map cannot be used after
    /// calling this.
    #[inline]
    pub fn move_iter(self) -> MoveEntries<K, V> {
        unsafe {
            let mself: &mut HashMap<K, V> = transmute(&self);

            // Make Drop safe to use by essentially setting self.capacity()
            // to zero.
            let hashes = util::replace(&mut mself.hashes, make_uninitialized_vec(0));
            let keys   = util::replace(&mut mself.keys,   make_uninitialized_vec(0));
            let vals   = util::replace(&mut mself.vals,   make_uninitialized_vec(0));
            let size   = util::replace(&mut mself.size,   0);

            MoveEntries { hashes:    hashes
                        , keys:      keys
                        , vals:      vals
                        , idx:       0
                        , size_hint: size
                        }
        }
    }

/* Enable this code block to print the hashtable to stdout.
    /// Missing doc.
    pub fn debug_show(&self) {
        print!("[ ");

        if self.get_hash(0) == EMPTY_BUCKET {
            print!("HOLE");
        } else {
            print!("({:?}, {:?}, {:?})", self.get_hash(0), self.get_key_ref(0), self.get_val_ref(0));
        }

        for i in range(0, self.capacity()) {
            if self.get_hash(i) == EMPTY_BUCKET  {
                print!(", HOLE");
            } else {
                print!(", ({:?}, {:?}, {:?})", self.get_hash(i), self.get_key_ref(i), self.get_val_ref(i));
            }
        }
        print!(" ]\n");
    }
*/
}

impl<K: Hash + Eq, V: Clone> HashMap<K, V> {
    /// Like `find`, but returns a copy of the value.
    #[inline]
    pub fn find_copy(&self, k: &K) -> Option<V> {
        self.find(k).map(|v| (*v).clone())
    }

    /// Deprecated. Use 'expect_copy'.
    pub fn get_copy(&self, k: &K) -> V {
        (*self.expect(k)).clone()
    }

    /// Like `expect`, but returns a copy of the value.
    #[inline]
    pub fn expect_copy(&self, k: &K) -> V {
        (self.expect(k)).clone()
    }
}

impl<K: Hash + Eq, V: Eq> Eq for HashMap<K, V> {
    fn eq(&self, other: &HashMap<K, V>) -> bool {
        if self.len() != other.len() { return false; }

        self.iter().all(|(key, value)| {
            match other.find(key) {
                None    => false,
                Some(v) => value == v
            }
        })
    }

    #[inline]
    fn ne(&self, other: &HashMap<K, V>) -> bool { !self.eq(other) }
}

impl<K: Hash + Eq + Clone, V: Clone> Clone for HashMap<K, V> {
    fn clone(&self) -> HashMap<K, V> {
        let mut new_map = HashMap::with_capacity(self.capacity());

        for i in range(0, self.capacity()) {
            let hash = self.get_hash(i);

            if hash == EMPTY_BUCKET { continue }
            new_map.manual_insert_hashed(
                hash,
                (*self.get_key_ref(i)).clone(),
                (*self.get_val_ref(i)).clone());
        }

        new_map
    }
}

impl <K: fmt::Show + Eq + Hash, V: fmt::Show> fmt::Show for HashMap<K, V> {
    fn fmt(m: &HashMap<K, V>, f: &mut fmt::Formatter) -> fmt::Result {
        if m.is_empty() {
            if_ok!(f.buf.write_str("{}"));
        }

        let capacity = m.hashes.len();
        let mut first_elem_idx = capacity;

        for i in range(0, capacity) {
            if m.get_hash(i) == EMPTY_BUCKET { continue }
            first_elem_idx = i;
            break;
        }

        // we know it has at least one element, so make the first one.
        if_ok!(write!(f.buf, "{}", "{ "));
        if_ok!(write!(f.buf, "({}, {})",
            *m.get_key_ref(first_elem_idx),
            *m.get_val_ref(first_elem_idx)));

        for i in range(first_elem_idx + 1, capacity) {
            if m.get_hash(i) == EMPTY_BUCKET { continue }
            if_ok!(write!(f.buf, ", "));
            if_ok!(write!(f.buf, "({}, {})",
                *m.get_key_ref(i),
                *m.get_val_ref(i)));
        }

        write!(f.buf, "{}", if m.size == 1 { "}" } else { " }" })
    }
}

impl<K: Eq + Hash, V> Default for HashMap<K, V> {
    #[inline]
    fn default() -> HashMap<K, V> { HashMap::new() }
}

/// HashMap iterator
#[deriving(Clone)]
pub struct Entries<'a, K, V> {
    priv hashes:    &'a vec_ng::Vec<u64>,
    priv keys:      &'a vec_ng::Vec<K>,
    priv vals:      &'a vec_ng::Vec<V>,
    priv idx:       uint,
    priv size_hint: uint,
}

/// HashMap mutable values iterator
pub struct MutEntries<'a, K, V> {
    priv hashes: &'a     vec_ng::Vec<u64>,
    priv keys:   &'a     vec_ng::Vec<K>,
    priv vals:   &'a mut vec_ng::Vec<V>,
    priv idx:            uint,
    priv size_hint:      uint
}

/// HashMap move iterator
pub struct MoveEntries<K, V> {
    priv hashes:    vec_ng::Vec<u64>,
    priv keys:      vec_ng::Vec<K>,
    priv vals:      vec_ng::Vec<V>,
    priv idx:       uint,
    priv size_hint: uint,
}

/// HashMap keys iterator
pub type Keys<'a, K, V> =
    iter::Map<'static, (&'a K, &'a V), &'a K, Entries<'a, K, V>>;

/// HashMap values iterator
pub type Values<'a, K, V> =
    iter::Map<'static, (&'a K, &'a V), &'a V, Entries<'a, K, V>>;

impl<'a, K: Hash + Eq, V> Iterator<(&'a K, &'a V)> for Entries<'a, K, V> {
    #[inline]
    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        while self.idx < self.hashes.len() {
            let i = self.idx;
            self.idx += 1;

            if *get_vec_ref(self.hashes, i) == EMPTY_BUCKET { continue }

            self.size_hint -= 1;

            return Some((get_vec_ref(self.keys, i),
                         get_vec_ref(self.vals, i)));
        }

        None
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        (self.size_hint, Some(self.size_hint))
    }
}

impl<'a, K: Hash + Eq, V> Iterator<(&'a K, &'a mut V)> for MutEntries<'a, K, V> {
    #[inline]
    fn next(&mut self) -> Option<(&'a K, &'a mut V)> {
        while self.idx < self.hashes.len() {
            let i = self.idx;
            self.idx += 1;

            if *get_vec_ref(self.hashes, i) == EMPTY_BUCKET { continue }

            // FIXME: I don't know why I need to transmute the mutable value,
            // but I do! There's likely a bug here.
            unsafe {
                let k: &'a     K = get_vec_ref(self.keys, i);
                let v: &'a mut V = transmute(get_vec_mut_ref(self.vals, i));

                self.size_hint -= 1;

                return Some((k, v))
            }
        }

        None
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        (self.size_hint, Some(self.size_hint))
    }
}

impl<K: Hash + Eq, V> Iterator<(K, V)> for MoveEntries<K, V> {
    #[inline]
    fn next(&mut self) -> Option<(K, V)> {
        while self.idx < self.hashes.len() {
            let i = self.idx;
            self.idx += 1;

            if *get_vec_ref(&self.hashes, i) == EMPTY_BUCKET { continue }

            *get_vec_mut_ref(&mut self.hashes, i) = EMPTY_BUCKET;
            let k = take_vec_elem(&mut self.keys, i);
            let v = take_vec_elem(&mut self.vals, i);
            self.size_hint -= 1;

            return Some((k, v))
        }

        return None
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        (self.size_hint, Some(self.size_hint))
    }
}

impl<K: Eq + Hash, V> FromIterator<(K, V)> for HashMap<K, V> {
    #[inline]
    fn from_iterator<T: Iterator<(K, V)>>(iter: &mut T) -> HashMap<K, V> {
        let (lower, _) = iter.size_hint();
        let mut map = HashMap::with_capacity(lower);
        map.extend(iter);
        map
    }
}

impl<K: Eq + Hash, V> Extendable<(K, V)> for HashMap<K, V> {
    #[inline]
    fn extend<T: Iterator<(K, V)>>(&mut self, iter: &mut T) {
        for (k, v) in *iter {
            self.insert(k, v);
        }
    }
}

#[unsafe_destructor]
impl <K, V> Drop for MoveEntries<K, V> {
    fn drop(&mut self) {
        // Run destructors manually on 'full' buckets. Skip the ones
        // we already moved out of.
        for i in range(self.idx, self.hashes.len()) {
            if *get_vec_ref(&self.hashes, i) == EMPTY_BUCKET { continue }

            *get_vec_mut_ref(&mut self.hashes, i) = EMPTY_BUCKET;
            take_vec_elem(   &mut self.keys, i);
            take_vec_elem(   &mut self.vals, i);
        }

        // Don't let destructors run twice! Without these, the vector destructor
        // will try to destruct all the elements (which are now uninitalized),
        // and we will be very sad.
        unsafe {
            self.hashes.set_len(0);
            self.keys.set_len(0);
            self.vals.set_len(0);
        }
    }
}

/// HashSet iterator
#[deriving(Clone)]
pub struct SetItems<'a, K> {
    priv iter: Entries<'a, K, ()>,
}

/// HashSet move iterator
pub struct SetMoveItems<K> {
    priv iter: MoveEntries<K, ()>,
}

impl<'a, K: Hash + Eq> Iterator<&'a K> for SetItems<'a, K> {
    #[inline]
    fn next(&mut self) -> Option<&'a K> {
        match self.iter.next() {
            Some((k, _)) => Some(k),
            None         => None
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        self.iter.size_hint()
    }
}

impl<K: Hash + Eq> Iterator<K> for SetMoveItems<K> {
    #[inline]
    fn next(&mut self) -> Option<K> {
        match self.iter.next() {
            Some((k, _)) => Some(k),
            None       => None
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        self.iter.size_hint()
    }
}

#[unsafe_destructor]
impl<K, V> Drop for HashMap<K, V> {
    fn drop(&mut self) {
        // Run destructors manually on 'full' buckets.
        for i in range(0, self.hashes.len()) {
            if *get_vec_ref(&self.hashes, i) == EMPTY_BUCKET { continue }

            *get_vec_mut_ref(&mut self.hashes, i) = EMPTY_BUCKET;
            take_vec_elem(   &mut self.keys, i);
            take_vec_elem(   &mut self.vals, i);

            self.size -= 1;
        }

        // Don't let destructors run twice! Without these, the vector destructor
        // will try to destruct all the elements (which are now uninitalized),
        // and we will be very sad.
        unsafe {
            self.hashes.set_len(0);
            self.keys.set_len(0);
            self.vals.set_len(0);
        }
    }
}


/// An implementation of a hash set using the underlying representation of a
/// HashMap where the value is (). As with the `HashMap` type, a `HashSet`
/// requires that the elements implement the `Eq` and `Hash` traits.
pub struct HashSet<T> {
    priv map: HashMap<T, ()>
}

impl<T:Hash + Eq> Eq for HashSet<T> {
    // I would normally delegate this to the underlying hashmap,
    // but can't due to #11998.
    #[inline]
    fn eq(&self, other: &HashSet<T>) -> bool {
        if self.map.len() != other.map.len() { return false }
        
        self.map.iter().all(|(k, _)| {
            other.map.search(k).is_some()
        })
    }

    #[inline]
    fn ne(&self, other: &HashSet<T>) -> bool { self.map != other.map }
}

impl<T: Hash + Eq> Container for HashSet<T> {
    /// Return the number of elements in the set
    #[inline]
    fn len(&self) -> uint { self.map.len() }
}

impl<T:Hash + Eq> Mutable for HashSet<T> {
    /// Clear the set, removing all values.
    #[inline]
    fn clear(&mut self) { self.map.clear() }
}

impl<T: Hash + Eq> Set<T> for HashSet<T> {
    /// Return true if the set contains a value
    #[inline]
    fn contains(&self, value: &T) -> bool { self.map.search(value).is_some() }

    /// Return true if the set has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    #[inline]
    fn is_disjoint(&self, other: &HashSet<T>) -> bool {
        self.iter().all(|v| !other.contains(v))
    }

    /// Return true if the set is a subset of another
    #[inline]
    fn is_subset(&self, other: &HashSet<T>) -> bool {
        self.iter().all(|v| other.contains(v))
    }

    /// Return true if the set is a superset of another
    #[inline]
    fn is_superset(&self, other: &HashSet<T>) -> bool {
        other.is_subset(self)
    }
}

impl<T: Hash + Eq> MutableSet<T> for HashSet<T> {
    /// Add a value to the set. Return true if the value was not already
    /// present in the set.
    #[inline]
    fn insert(&mut self, value: T) -> bool { self.map.insert(value, ()) }

    /// Remove a value from the set. Return true if the value was
    /// present in the set.
    #[inline]
    fn remove(&mut self, value: &T) -> bool { self.map.remove(value) }
}

impl<T: Hash + Eq> HashSet<T> {
    /// Create an empty HashSet
    #[inline]
    pub fn new() -> HashSet<T> {
        HashSet::with_capacity(INITIAL_CAPACITY)
    }

/* For debugging.
    /// No docs.
    pub fn debug_show(&self) {
        self.map.debug_show();
    }
*/

    /// Create an empty HashSet with space for at least `n` elements in
    /// the hash table.
    #[inline]
    pub fn with_capacity(capacity: uint) -> HashSet<T> {
        HashSet { map: HashMap::with_capacity(capacity) }
    }

    /// Create an empty HashSet with space for at least `capacity`
    /// elements in the hash table, using `k0` and `k1` as the keys.
    ///
    /// Warning: `k0` and `k1` are normally randomly generated, and
    /// are designed to allow HashSets to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting them
    /// manually using this function can expose a DoS attack vector.
    #[inline]
    pub fn with_capacity_and_keys(k0: u64, k1: u64, capacity: uint) -> HashSet<T> {
        HashSet { map: HashMap::with_capacity_and_keys(k0, k1, capacity) }
    }

    /// Reserve space for at least `n` elements in the hash table.
    #[inline]
    pub fn reserve_at_least(&mut self, n: uint) {
        self.map.reserve_at_least(n)
    }

    /// Returns true if the hash set contains a value equivalent to the
    /// given query value.
    #[inline]
    pub fn contains_equiv<Q:Hash + Equiv<T>>(&self, value: &Q) -> bool {
      self.map.contains_key_equiv(value)
    }

    /// An iterator visiting all elements in arbitrary order.
    /// Iterator element type is &'a T.
    #[inline]
    pub fn iter<'a>(&'a self) -> SetItems<'a, T> {
        SetItems { iter: self.map.iter() }
    }

    /// Creates a consuming iterator, that is, one that moves each value out
    /// of the set in arbitrary order. The set cannot be used after calling
    /// this.
    #[inline]
    pub fn move_iter(self) -> SetMoveItems<T> {
        SetMoveItems {iter: self.map.move_iter()}
    }

    /// Visit the values representing the difference
    #[inline]
    pub fn difference<'a>(&'a self, other: &'a HashSet<T>) -> SetAlgebraItems<'a, T> {
        Repeat::new(other)
            .zip(self.iter())
            .filter_map(|(other, elt)| {
                if !other.contains(elt) { Some(elt) } else { None }
            })
    }

    /// Visit the values representing the symmetric difference
    #[inline]
    pub fn symmetric_difference<'a>(&'a self, other: &'a HashSet<T>)
        -> Chain<SetAlgebraItems<'a, T>, SetAlgebraItems<'a, T>> {
        self.difference(other).chain(other.difference(self))
    }

    /// Visit the values representing the intersection
    #[inline]
    pub fn intersection<'a>(&'a self, other: &'a HashSet<T>)
        -> SetAlgebraItems<'a, T> {
        Repeat::new(other)
            .zip(self.iter())
            .filter_map(|(other, elt)| {
                if other.contains(elt) { Some(elt) } else { None }
            })
    }

    /// Visit the values representing the union
    #[inline]
    pub fn union<'a>(&'a self, other: &'a HashSet<T>)
        -> Chain<SetItems<'a, T>, SetAlgebraItems<'a, T>> {
        self.iter().chain(other.difference(self))
    }

}

impl<T: Hash + Eq + Clone> Clone for HashSet<T> {
    #[inline]
    fn clone(&self) -> HashSet<T> {
        HashSet {
            map: self.map.clone()
        }
    }
}

impl<K: Eq + Hash> FromIterator<K> for HashSet<K> {
    fn from_iterator<T: Iterator<K>>(iter: &mut T) -> HashSet<K> {
        let (lower, _) = iter.size_hint();
        let mut set = HashSet::with_capacity(lower);
        set.extend(iter);
        set
    }
}

impl<K: Eq + Hash> Extendable<K> for HashSet<K> {
    fn extend<T: Iterator<K>>(&mut self, iter: &mut T) {
        for k in *iter {
            self.insert(k);
        }
    }
}

impl<K: Eq + Hash> Default for HashSet<K> {
    fn default() -> HashSet<K> { HashSet::new() }
}

// `Repeat` is used to feed the filter closure an explicit capture
// of a reference to the other set
/// Set operations iterator
pub type SetAlgebraItems<'a, T> =
    FilterMap<'static,(&'a HashSet<T>, &'a T), &'a T,
              Zip<Repeat<&'a HashSet<T>>,SetItems<'a,T>>>;

#[cfg(test)]
mod test_map {
    use prelude::*;
    use super::*;
    use iter::*;
    use container::*;

    #[test]
    fn test_create_capacity_zero() {
        let mut m = HashMap::with_capacity(0);

        assert!(m.insert(1, 1));

        assert!(m.contains_key(&1));
        assert!(!m.contains_key(&0));
    }

    #[test]
    fn test_insert() {
        let mut m = HashMap::new();
        assert_eq!(m.len(), 0);
        assert!(m.insert(1, 2));
        assert_eq!(m.len(), 1);
        assert!(m.insert(2, 4));
        assert_eq!(m.len(), 2);
        assert_eq!(*m.find(&1).unwrap(), 2);
        assert_eq!(*m.find(&2).unwrap(), 4);
    }

/* Destructors are hard to test. Uncomment this, and then run:

    > make check-stage1-std NO_BENCH=1 NO_REBUILD=1 TESTNAME=test_drops
    
   to run this test. You have to manually examine the output, and should
   get a -[n] for every +[n].

    struct Noisy {
        pub k: int
    }

    impl IterBytes for Noisy {
        fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
            self.k.iter_bytes(lsb0, f)
        }
    }

    impl Noisy {
        fn new(k: int) -> Noisy {
            println!("+{}", k);
            Noisy { k: k }
        }
    }

    impl Eq for Noisy {
        fn eq(&self, other: &Noisy) -> bool {
            self.k == other.k
        }
    }

    impl Drop for Noisy {
        fn drop(&mut self) {
            println!("-{}", self.k);
        }
    }

    #[test]
    fn test_drops() {
        {
            let mut m = HashMap::new();

            println!("starting");

            for i in range(0, 100) {
                println!("creating {} & {}", i, i+100);
                let n1 = Noisy::new(i);
                let n2 = Noisy::new(i+100);
                println!("  > created. inserting...");
                m.insert(n1, n2);
                println!("  > inserted.");
            }

            println!("Removing 0-49");

            for i in range(0, 50) {
                println!("Making noisy {} to pop...", i);
                let k = Noisy::new(i);
                println!("  > popping.");
                assert_eq!(m.pop(&k).unwrap().k, i + 100);
                println!("  > Done pop.")
            }

            println!("Destructing map.");
        }

        println!("Done destructing. Everything should be cleaned up by now.");
    }
*/

    #[test]
    fn test_empty_pop() {
        let mut m: HashMap<int, bool> = HashMap::new();
        assert_eq!(m.pop(&0), None);
    }

    #[test]
    fn test_lots_of_insertions() {
        let mut m = HashMap::new();

        // Try this a few times to make sure we never screw up the hashmap's
        // internal state.
        for _ in range(0, 10) {
            assert!(m.is_empty());

            for i in range_inclusive(1, 1000) {
                assert!(m.insert(i, i));

                for j in range_inclusive(1, i) {
                    let r = m.find(&j);
                    assert_eq!(r, Some(&j));
                }

                for j in range_inclusive(i+1, 1000) {
                    let r = m.find(&j);
                    assert_eq!(r, None);
                }
            }

            for i in range_inclusive(1001, 2000) {
                assert!(!m.contains_key(&i));
            }

            // remove forwards
            for i in range_inclusive(1, 1000) {
                assert!(m.remove(&i));

                for j in range_inclusive(1, i) {
                    assert!(!m.contains_key(&j));
                }

                for j in range_inclusive(i+1, 1000) {
                    assert!(m.contains_key(&j));
                }
            }

            for i in range_inclusive(1, 1000) {
                assert!(!m.contains_key(&i));
            }

            for i in range_inclusive(1, 1000) {
                assert!(m.insert(i, i));
            }

            // remove backwards
            for i in range_step_inclusive(1000, 1, -1) {
                assert!(m.remove(&i));

                for j in range_inclusive(i, 1000) {
                    assert!(!m.contains_key(&j));
                }

                for j in range_inclusive(1, i-1) {
                    assert!(m.contains_key(&j));
                }
            }
        }
    }

    #[test]
    fn test_find_mut() {
        let mut m = HashMap::new();
        assert!(m.insert(1, 12));
        assert!(m.insert(2, 8));
        assert!(m.insert(5, 14));
        let new = 100;
        match m.find_mut(&5) {
            None => fail!(), Some(x) => *x = new
        }
        assert_eq!(m.find(&5), Some(&new));
    }

    #[test]
    fn test_insert_overwrite() {
        let mut m = HashMap::new();
        assert!(m.insert(1, 2));
        assert_eq!(*m.find(&1).unwrap(), 2);
        assert!(!m.insert(1, 3));
        assert_eq!(*m.find(&1).unwrap(), 3);
    }

    #[test]
    fn test_insert_conflicts() {
        let mut m = HashMap::with_capacity(4);
        assert!(m.insert(1, 2));
        assert!(m.insert(5, 3));
        assert!(m.insert(9, 4));
        assert_eq!(*m.find(&9).unwrap(), 4);
        assert_eq!(*m.find(&5).unwrap(), 3);
        assert_eq!(*m.find(&1).unwrap(), 2);
    }

    #[test]
    fn test_conflict_remove() {
        let mut m = HashMap::with_capacity(4);
        assert!(m.insert(1, 2));
        assert!(m.insert(5, 3));
        assert!(m.insert(9, 4));
        assert!(m.remove(&1));
        assert_eq!(*m.find(&9).unwrap(), 4);
        assert_eq!(*m.find(&5).unwrap(), 3);
    }

    #[test]
    fn test_is_empty() {
        let mut m = HashMap::with_capacity(4);
        assert!(m.insert(1, 2));
        assert!(!m.is_empty());
        assert!(m.remove(&1));
        assert!(m.is_empty());
    }

    #[test]
    fn test_pop() {
        let mut m = HashMap::new();
        m.insert(1, 2);
        assert_eq!(m.pop(&1), Some(2));
        assert_eq!(m.pop(&1), None);
    }

    #[test]
    fn test_swap() {
        let mut m = HashMap::new();
        assert_eq!(m.swap(1, 2), None);
        assert_eq!(m.swap(1, 3), Some(2));
        assert_eq!(m.swap(1, 4), Some(3));
    }

    #[test]
    fn test_move_iter() {
        let hm = {
            let mut hm = HashMap::new();

            hm.insert('a', 1);
            hm.insert('b', 2);

            hm
        };

        let v = hm.move_iter().collect::<~[(char, int)]>();
        assert!([('a', 1), ('b', 2)] == v || [('b', 2), ('a', 1)] == v);
    }

    #[test]
    fn test_iterate() {
        let mut m = HashMap::with_capacity(4);
        for i in range(0u, 32) {
            assert!(m.insert(i, i*2));
        }
        assert_eq!(m.len(), 32);

        let mut observed = 0;

        for (k, v) in m.iter() {
            assert_eq!(*v, *k * 2);
            observed |= (1 << *k);
        }
        assert_eq!(observed, 0xFFFF_FFFF);
    }

    #[test]
    fn test_keys() {
        let vec = ~[(1, 'a'), (2, 'b'), (3, 'c')];
        let map = vec.move_iter().collect::<HashMap<int, char>>();
        let keys = map.keys().map(|&k| k).collect::<~[int]>();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn test_values() {
        let vec = ~[(1, 'a'), (2, 'b'), (3, 'c')];
        let map = vec.move_iter().collect::<HashMap<int, char>>();
        let values = map.values().map(|&v| v).collect::<~[char]>();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&'a'));
        assert!(values.contains(&'b'));
        assert!(values.contains(&'c'));
    }

    #[test]
    fn test_find() {
        let mut m = HashMap::new();
        assert!(m.find(&1).is_none());
        m.insert(1, 2);
        match m.find(&1) {
            None => fail!(),
            Some(v) => assert!(*v == 2)
        }
    }

    #[test]
    fn test_eq() {
        let mut m1 = HashMap::new();
        m1.insert(1, 2);
        m1.insert(2, 3);
        m1.insert(3, 4);

        let mut m2 = HashMap::new();
        m2.insert(1, 2);
        m2.insert(2, 3);

        assert!(m1 != m2);

        m2.insert(3, 4);

        assert_eq!(m1, m2);
    }

    #[test]
    fn test_expand() {
        let mut m = HashMap::new();

        assert_eq!(m.len(), 0);
        assert!(m.is_empty());

        let mut i = 0u;
        let old_resize_at = m.grow_at;
        while old_resize_at == m.grow_at {
            m.insert(i, i);
            i += 1;
        }

        assert_eq!(m.len(), i);
        assert!(!m.is_empty());
    }

    #[test]
    fn test_find_equiv() {
        let mut m = HashMap::new();

        let (foo, bar, baz) = (1,2,3);
        m.insert(~"foo", foo);
        m.insert(~"bar", bar);
        m.insert(~"baz", baz);


        assert_eq!(m.find_equiv(&("foo")), Some(&foo));
        assert_eq!(m.find_equiv(&("bar")), Some(&bar));
        assert_eq!(m.find_equiv(&("baz")), Some(&baz));

        assert_eq!(m.find_equiv(&("qux")), None);
    }

    #[test]
    fn test_from_iter() {
        let xs = ~[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: HashMap<int, int> = xs.iter().map(|&x| x).collect();

        for &(k, v) in xs.iter() {
            assert_eq!(map.find(&k), Some(&v));
        }
    }
}

#[cfg(test)]
mod test_set {
    use super::*;
    use prelude::*;
    use container::Container;
    use vec::ImmutableEqVector;

    #[test]
    fn test_disjoint() {
        let mut xs = HashSet::new();
        let mut ys = HashSet::new();
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(xs.insert(5));
        assert!(ys.insert(11));
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(xs.insert(7));
        assert!(xs.insert(19));
        assert!(xs.insert(4));
        assert!(ys.insert(2));
        assert!(ys.insert(-11));
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(ys.insert(7));
        assert!(!xs.is_disjoint(&ys));
        assert!(!ys.is_disjoint(&xs));
    }

    #[test]
    fn test_subset_and_superset() {
        let mut a = HashSet::new();
        assert!(a.insert(0));
        assert!(a.insert(5));
        assert!(a.insert(11));
        assert!(a.insert(7));

        let mut b = HashSet::new();
        assert!(b.insert(0));
        assert!(b.insert(7));
        assert!(b.insert(19));
        assert!(b.insert(250));
        assert!(b.insert(11));
        assert!(b.insert(200));

        assert!(!a.is_subset(&b));
        assert!(!a.is_superset(&b));
        assert!(!b.is_subset(&a));
        assert!(!b.is_superset(&a));

        assert!(b.insert(5));

        assert!(a.is_subset(&b));
        assert!(!a.is_superset(&b));
        assert!(!b.is_subset(&a));
        assert!(b.is_superset(&a));
    }

    #[test]
    fn test_iterate() {
        let mut a = HashSet::new();
        for i in range(0u, 32) {
            assert!(a.insert(i));
        }
        let mut observed = 0;
        for k in a.iter() {
            observed |= (1 << *k);
        }
        assert_eq!(observed, 0xFFFF_FFFF);
    }

    #[test]
    fn test_intersection() {
        let mut a = HashSet::new();
        let mut b = HashSet::new();

        assert!(a.insert(11));
        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(77));
        assert!(a.insert(103));
        assert!(a.insert(5));
        assert!(a.insert(-5));

        assert!(b.insert(2));
        assert!(b.insert(11));
        assert!(b.insert(77));
        assert!(b.insert(-9));
        assert!(b.insert(-42));
        assert!(b.insert(5));
        assert!(b.insert(3));

        let mut i = 0;
        let expected = [3, 5, 11, 77];
        for x in a.intersection(&b) {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_difference() {
        let mut a = HashSet::new();
        let mut b = HashSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));

        assert!(b.insert(3));
        assert!(b.insert(9));

        let mut i = 0;
        let expected = [1, 5, 11];
        for x in a.difference(&b) {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_symmetric_difference() {
        let mut a = HashSet::new();
        let mut b = HashSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));

        assert!(b.insert(-2));
        assert!(b.insert(3));
        assert!(b.insert(9));
        assert!(b.insert(14));
        assert!(b.insert(22));

        let mut i = 0;
        let expected = [-2, 1, 5, 11, 14, 22];
        for x in a.symmetric_difference(&b) {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_union() {
        let mut a = HashSet::new();
        let mut b = HashSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));
        assert!(a.insert(16));
        assert!(a.insert(19));
        assert!(a.insert(24));

        assert!(b.insert(-2));
        assert!(b.insert(1));
        assert!(b.insert(5));
        assert!(b.insert(9));
        assert!(b.insert(13));
        assert!(b.insert(19));

        let mut i = 0;
        let expected = [-2, 1, 3, 5, 9, 11, 13, 16, 19, 24];
        for x in a.union(&b) {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_from_iter() {
        let xs = ~[1, 2, 3, 4, 5, 6, 7, 8, 9];

        let set: HashSet<int> = xs.iter().map(|&x| x).collect();

        for x in xs.iter() {
            assert!(set.contains(x));
        }
    }

    #[test]
    fn test_move_iter() {
        let hs = {
            let mut hs = HashSet::new();

            hs.insert('a');
            hs.insert('b');

            hs
        };

        let v = hs.move_iter().collect::<~[char]>();
        assert!(['a', 'b'] == v || ['b', 'a'] == v);
    }

    #[test]
    fn test_eq() {
        let mut s1 = HashSet::with_capacity_and_keys(15548002063724546226u64,
                                                     11698216626079898609u64, 32u);
        s1.insert(1);
        s1.insert(2);
        s1.insert(3);

        let mut s2 = HashSet::with_capacity_and_keys(1266625446389752373u64,
                                                     1391914631781420694u64, 32u);

        s2.insert(1);
        s2.insert(2);

        assert!(s1 != s2);

        s2.insert(3);

        assert_eq!(s1, s2);
    }
}

/*
#[cfg(test)]
mod bench {
    use extra::test::BenchHarness;
    use iter::*;
    use option::*;
    use prelude::*;

    use hashmap_legacy;

    #[bench]
    fn old_hashmap_insert(b: &mut BenchHarness) {
        let mut m: hashmap_legacy::HashMap<int, int> = hashmap_legacy::HashMap::new();

        for i in range_inclusive(1, 1000) {
            m.insert(i, i);
        }

        let mut k = 1001;

        b.iter(|| {
            m.insert(k, k);
            k += 1;
        });
    }

    #[bench]
    fn new_hashmap_insert(b: &mut BenchHarness) {
        use super::*;

        let mut m = HashMap::new();

        for i in range_inclusive(1, 1000) {
            m.insert(i, i);
        }

        let mut k = 1001;

        b.iter(|| {
            m.insert(k, k);
            k += 1;
        });
    }

    #[bench]
    fn old_hashmap_find_existing(b: &mut BenchHarness) {
        let mut m: hashmap_legacy::HashMap<int, int> = hashmap_legacy::HashMap::new();

        for i in range_inclusive(1, 1000) {
            m.insert(i, i);
        }

        b.iter(|| {
            m.contains_key(&412);
        });
    }

    #[bench]
    fn new_hashmap_find_existing(b: &mut BenchHarness) {
        use super::*;

        let mut m = HashMap::new();

        for i in range_inclusive(1, 1000) {
            m.insert(i, i);
        }

        b.iter(|| {
            m.contains_key(&412);
        });
    }

    #[bench]
    fn old_hashmap_find_notexisting(b: &mut BenchHarness) {
        let mut m: hashmap_legacy::HashMap<int, int> = hashmap_legacy::HashMap::new();

        for i in range_inclusive(1, 1000) {
            m.insert(i, i);
        }

        b.iter(|| {
            m.contains_key(&2048);
        });
    }

    #[bench]
    fn new_hashmap_find_notexisting(b: &mut BenchHarness) {
        use super::*;

        let mut m = HashMap::new();

        for i in range_inclusive(1, 1000) {
            m.insert(i, i);
        }

        b.iter(|| {
            m.contains_key(&2048);
        });
    }

    #[bench]
    fn old_hashmap_as_queue(b: &mut BenchHarness) {
        let mut m: hashmap_legacy::HashMap<int, int> = hashmap_legacy::HashMap::new();

        for i in range_inclusive(1, 1000) {
            m.insert(i, i);
        }

        let mut k = 1;

        b.iter(|| {
            m.pop(&k);
            m.insert(k + 1000, k + 1000);
            k += 1;
        })
    }

    #[bench]
    fn new_hashmap_as_queue(b: &mut BenchHarness) {
        use super::*;

        let mut m = HashMap::new();

        for i in range_inclusive(1, 1000) {
            m.insert(i, i);
        }

        let mut k = 1;

        b.iter(|| {
            m.pop(&k);
            m.insert(k + 1000, k + 1000);
            k += 1;
        });
    }

    #[bench]
    fn comprehensive_old_hashmap(b: &mut BenchHarness) {
        let mut m: hashmap_legacy::HashMap<int, int> = hashmap_legacy::HashMap::new();

        for i in range_inclusive(1, 1000) {
            m.insert(i, i);
        }

        let mut k = 1;

        b.iter(|| {
            m.find(&(k + 400));
            m.find(&(k + 2000));
            m.pop(&k);
            m.insert(k + 1000, k + 1000);
            k += 1;
        });
    }

    #[bench]
    fn comprehensive_new_hashmap(b: &mut BenchHarness) {
        use super::*;

        let mut m = HashMap::new();

        for i in range_inclusive(1, 1000) {
            m.insert(i, i);
        }

        let mut k = 1;

        b.iter(|| {
            m.find(&(k + 400));
            m.find(&(k + 2000));
            m.pop(&k);
            m.insert(k + 1000, k + 1000);
            k += 1;
        })
    }
}
*/
