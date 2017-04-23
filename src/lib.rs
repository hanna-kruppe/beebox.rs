//! Yet another axis-aligned bounding box (AABB), biased towards ray tracing.
//!
//! The design is loosely based on the `BBox` class from the second edition of *Physically
//! based rendering: From theory to implementation* by Matt Pharr and Greg Humphreys.
//! Consequently, it is geared to the demands of ray tracing and similar applications, and comes
//! with otherwise rarely-used operations such as surface area calculations and ray-AABB
//! intersection calculations.
//!
//! It is also focused entirely on 32-bit floats.
//! A 64-bit variant may be added in later versions if needed.

pub extern crate cgmath;

use std::f32;
use cgmath::{Vector3, vec3};

// TODO some operations might be easier to SIMD-fy with this layout: struct Aabb([f32; 6]);

/// An axis-aligned bounding box in 3D space.
///
/// The box can, for the most part, be thought of as being defined by two corners,
/// a minimum corner and a maximum corner.
/// All coordinates are 32-bit floats.
///
/// This representation does not allow empty boxes, which are occasionally useful.
/// However, one can abuse the corner vectors and set the "minimum corner" to *positive* infinity
/// and the "maximum corner" to *negative* infinity.
/// Most AABB operations will handle this without extra work if they're written carefully, but it's
/// an unnatural special case to keep in mind when working with the corners.
///
/// `Aabb` can be empty, which is implemented using the aforementioned trick, but as far as the
/// interface is concerned, it upholds the "min corner, max corner" view: empty `Aabb`s cannot be
/// constructed by specifying nonsensical corners (there is a separate constructor, `empty()`), and
/// there is no minimum or maximum corner for an empty bounding box (the accessors panic).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Aabb {
    min: Vector3<f32>,
    max: Vector3<f32>,
}

impl Aabb {
    /// Constructs a bounding box from a set of points.
    ///
    /// Empty sets of points are valid and result in an empty bounding box.
    pub fn new<I>(points: I) -> Self
        where I: IntoIterator<Item = Vector3<f32>>
    {
        let mut res = Aabb::empty();
        for p in points {
            res.add_point(p);
        }
        res
    }

    /// Returns the "minimum corner", i.e., the coordinate-wise minimum of the region of space
    /// described by the AABB.
    ///
    /// # Panics
    ///
    /// Empty AABBs have no meaningful minimum, so calling this functions on an empty AABB panics.
    pub fn min(&self) -> Vector3<f32> {
        assert!(!self.is_empty(), "Empty AABB has no minimum corner");
        self.min
    }

    /// Returns the "maximum corner", i.e., the coordinate-wise maximum of the region of space
    /// described by the AABB.
    ///
    /// # Panics
    ///
    /// Empty AABBs have no meaningful maximum, so calling this functions on an empty AABB panics.
    pub fn max(&self) -> Vector3<f32> {
        assert!(!self.is_empty(), "Empty AABB has no maximum corner");
        self.max
    }

    /// Constructs a bounding box by supplying the minimum and maximum coordinates directly.
    ///
    /// Empty AABBs cannot be created through this function, consider using `Aabb::empty()`.
    ///
    /// # Panics
    ///
    /// Panics when the corners aren't really minimum and maximum, i.e. `min[i] > max[i]` in
    /// any dimension `i`.
    pub fn from_corners(min: Vector3<f32>, max: Vector3<f32>) -> Self {
        assert!(min.x <= max.x && min.y <= max.y && min.z <= max.z,
                "max must not be greater than min");
        Aabb {
            min: min,
            max: max,
        }
    }

    /// Constructs an empty AABB which does not contain any points and can be `union`'d with
    /// other AABBs without affecting them.
    pub fn empty() -> Self {
        let inf = vec3(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        Aabb {
            min: inf,
            max: -inf,
        }
    }

    /// Returns `true` if the box is empty, i.e., encompasses no space and not even a single point.
    pub fn is_empty(&self) -> bool {
        let inf = vec3(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        self.min == inf && self.max == -inf
    }

    /// Expand the bounding box to also encompass the point `p`.
    pub fn add_point(&mut self, p: Vector3<f32>) {
        // FIXME f32::min calls fmin, which is robust against NaN but may be
        // unnecessarily slow since it can't be mapped to SSE
        self.min.x = self.min.x.min(p.x);
        self.min.y = self.min.y.min(p.y);
        self.min.z = self.min.z.min(p.z);
        self.max.x = self.max.x.max(p.x);
        self.max.y = self.max.y.max(p.y);
        self.max.z = self.max.z.max(p.z);
    }

    /// Updates the AABB to also encompass `other`.
    /// This is an in-place version of `union`.
    pub fn add_box(&mut self, other: Self) {
        self.min.x = self.min.x.min(other.min.x);
        self.min.y = self.min.y.min(other.min.y);
        self.min.z = self.min.z.min(other.min.z);
        self.max.x = self.max.x.max(other.max.x);
        self.max.y = self.max.y.max(other.max.y);
        self.max.z = self.max.z.max(other.max.z);
    }

    /// Returns the smallest AABB that encompasses both `self` and `other`.
    pub fn union(&self, other: Self) -> Self {
        let mut bb = *self;
        bb.add_box(other);
        bb
    }

    /// Returns the surface area of the box.
    /// This is useful for construction of bounding volume hierarchies.
    ///
    /// # Panics
    ///
    /// Panics if the box is empty.
    /// While it might be possible to meaningfully assign surface area 0 to a non-existing
    /// surface, it's not 100% clear whether this is mathematically kosher and in the context of
    /// BVH construction, needing the surface area of an empty box probably indicates a bug.
    pub fn surface_area(&self) -> f32 {
        assert!(!self.is_empty(), "empty box has no surface area");
        let d = self.max - self.min;
        let area = 2.0 * (d.x * d.y + d.x * d.z + d.y * d.z);
        debug_assert!(area.is_finite());
        area
    }

    /// Returns `true` if the `[t0, t1]` segment of the ray intersects this AABB.
    /// In other words, returns `true` if the ray intersects the AABB and the intersection point
    /// is `r(t) := o + t * d` for `t` in the interval `[t0, t1]`.
    /// (As usual, `o` denotes the ray origin and `d` the ray direction.)
    ///
    /// The code is derived from *Physically based rendering* (cited in the crate documentation)
    /// and implements the algorithm from:
    ///
    /// > Williams, Amy, et al. "An efficient and robust ray-box intersection algorithm."
    /// > ACM SIGGRAPH 2005 Courses.
    pub fn intersects(&self, r: &RayData, t0: f32, t1: f32) -> bool {
        let p = [self.min, self.max];
        let (sgn0, sgn1, sgn2) = (r.sign[0] as usize, r.sign[1] as usize, r.sign[2] as usize);
        let mut tmin = (p[sgn0].x - r.org.x) * r.inv_dir.x;
        let mut tmax = (p[1 - sgn0].x - r.org.x) * r.inv_dir.x;
        let tymin = (p[sgn1].y - r.org.y) * r.inv_dir.y;
        let tymax = (p[1 - sgn1].y - r.org.y) * r.inv_dir.y;
        if tmin > tymax || tymin > tmax {
            return false;
        }
        if tymin > tmin {
            tmin = tymin;
        }
        if tymax < tmax {
            tmax = tymax;
        }
        let tzmin = (p[sgn2].z - r.org.z) * r.inv_dir.z;
        let tzmax = (p[1 - sgn2].z - r.org.z) * r.inv_dir.z;
        if tmin > tzmax || tzmin > tmax {
            return false;
        }
        if tzmin > tmin {
            tmin = tzmin;
        }
        if tzmax < tmax {
            tmax = tzmax;
        }
        tmin < t1 && tmax > t0
    }

    /// Returns the center point of the box.
    ///
    /// # Panics
    ///
    /// Panics if the box is empty.
    pub fn centroid(&self) -> Vector3<f32> {
        assert!(!self.is_empty(), "Empty box has no centroid");
        self.min * 0.5 + self.max * 0.5
    }

    /// Returns the axis along which the AABB is largest.
    ///
    /// In case multiple axes are largest (e.g., because the AABB is empty or a cube), the result is
    /// one of the largest axes, but it is unspecified which one is returned.
    pub fn largest_axis(&self) -> usize {
        let d = self.max - self.min;
        if d.x > d.y && d.x > d.z {
            0
        } else if d.y > d.z {
            1
        } else {
            2
        }
    }
}

/// Precomputed data for ray-AABB intersection calculation.
/// The data computed depends only on the ray and can be used to intersect one ray against
/// many different `Aabb`s.
pub struct RayData {
    sign: [u8; 3],
    org: Vector3<f32>,
    inv_dir: Vector3<f32>,
}

impl RayData {
    /// Prepares data for intersection calculation from ray origin `org` and ray direction
    /// `dir`.
    pub fn new(org: Vector3<f32>, dir: Vector3<f32>) -> Self {
        RayData {
            sign: [(dir.x < 0.0) as u8, (dir.y < 0.0) as u8, (dir.z < 0.0) as u8],
            org: org,
            inv_dir: 1.0 / dir,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f32;
    use cgmath::{Vector3, vec3};
    use cgmath::prelude::*;
    use super::{Aabb, RayData};

    #[test]
    fn new() {
        let bb = Aabb::new(vec![vec3(1.0, 0.5, -0.8), vec3(-1.0, 0.3, 0.0), vec3(0.7, 0.0, -0.4)]);
        assert_eq!(bb.min(), vec3(-1.0, 0.0, -0.8));
        assert_eq!(bb.max(), vec3(1.0, 0.5, 0.0));
    }

    #[test]
    fn new_empty() {
        let bb = Aabb::new(vec![]);
        assert!(bb.is_empty());
    }

    #[test]
    fn corners_roundtrip() {
        let bb = Aabb::from_corners(vec3(1.0, 2.0, 3.0), vec3(4.0, 5.0, 6.0));
        assert_eq!(bb.min(), vec3(1.0, 2.0, 3.0));
        assert_eq!(bb.max(), vec3(4.0, 5.0, 6.0));
    }

    #[test]
    fn is_empty() {
        let bb = Aabb::empty();
        assert!(bb.is_empty());
        let bb = Aabb::from_corners(Vector3::zero(), Vector3::zero());
        assert!(!bb.is_empty());
    }

    #[test]
    fn add_point() {
        let mut bb = Aabb::empty();
        let x = vec3(1.0, 2.0, 3.0);
        bb.add_point(x);
        assert_eq!(bb.min(), x);
        assert_eq!(bb.max(), x);
        bb.add_point(-x);
        assert_eq!(bb.min(), -x);
        assert_eq!(bb.max(), x);
        bb.add_point(vec3(1.0, 1.0, 1.5));
        assert_eq!(bb.min(), -x);
        assert_eq!(bb.max(), x);
        bb.add_point(vec3(0.0, 0.0, 4.0));
        assert_eq!(bb.min(), -x);
        assert_eq!(bb.max(), vec3(1.0, 2.0, 4.0));
    }

    #[test]
    fn add_box() {
        let mut bb = Aabb::empty();
        bb.add_box(Aabb::from_corners(vec3(-1.0, 0.0, 1.0), vec3(1.0, 0.2, 1.5)));
        assert_eq!(bb.min(), vec3(-1.0, 0.0, 1.0));
        assert_eq!(bb.max(), vec3(1.0, 0.2, 1.5));
        bb.add_box(Aabb::from_corners(vec3(3.0, -0.1, -2.0), vec3(4.0, 1.0, 0.0)));
        assert_eq!(bb.min(), vec3(-1.0, -0.1, -2.0));
        assert_eq!(bb.max(), vec3(4.0, 1.0, 1.5));
        bb.add_box(Aabb::empty());
        assert_eq!(bb.min(), vec3(-1.0, -0.1, -2.0));
        assert_eq!(bb.max(), vec3(4.0, 1.0, 1.5));
    }

    #[test]
    fn union() {
        let bb1 = Aabb::from_corners(vec3(-1.0, -2.0, -3.0), vec3(1.0, 1.0, 1.0));
        let bb2 = Aabb::from_corners(vec3(-2.0, -2.0, -3.0), vec3(2.0, 2.0, 2.0));
        assert_eq!(bb1.union(bb2), bb2);
        assert_eq!(bb2.union(bb1), bb2);
        let bb3 = Aabb::from_corners(vec3(5.0, 6.0, 3.0), vec3(6.0, 7.0, 8.0));
        let bb13 = Aabb::from_corners(vec3(-1.0, -2.0, -3.0), vec3(6.0, 7.0, 8.0));
        assert_eq!(bb1.union(bb3), bb13);
        assert_eq!(bb3.union(bb1), bb13);
    }

    #[test]
    fn empty_union() {
        let empty = Aabb::empty();
        let bb = Aabb::from_corners(vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0));
        assert_eq!(bb.union(empty), bb);
        assert_eq!(empty.union(bb), bb);
    }

    #[test]
    fn surface_area() {
        let bb = Aabb::from_corners(vec3(1.0, 0.5, 2.0), vec3(2.0, 1.0, 2.1));
        // The edge lengths are [1.0, 0.5, 0.1], and Wolfram Alpha computes a surface area of 1.3
        // from that.
        assert!((bb.surface_area() - 1.3).abs() < 1e-6);
        // Mirroring the AABB should not affect the surface area
        let bb2 = Aabb::from_corners(-bb.max(), -bb.min());
        assert_eq!(bb.surface_area(), bb2.surface_area());
    }

    #[test]
    fn intersects_inside() {
        let bb = Aabb::from_corners(vec3(-1.0, -2.0, -3.0), vec3(3.0, 2.0, 1.0));
        let ray = RayData::new(vec3(2.0, -1.0, 0.0), vec3(1.0, 0.0, 0.0));
        assert!(bb.intersects(&ray, 0.0, 1e-6));
    }

    #[test]
    fn intersects_outside() {
        let bb = Aabb::from_corners(vec3(-1.0, -2.0, -3.0), vec3(3.0, 2.0, 1.0));
        let ray = RayData::new(vec3(9.0, 9.0, 9.0), vec3(-0.33, -0.33, -0.33));
        assert!(bb.intersects(&ray, 0.0, 25.0));
    }

    #[test]
    fn intersects_respects_t0() {
        let bb = Aabb::from_corners(vec3(-1.0, -2.0, -3.0), vec3(3.0, 2.0, 1.0));
        let ray = RayData::new(vec3(9.0, 9.0, 9.0), vec3(-0.33, -0.33, -0.33));
        assert!(!bb.intersects(&ray, 50.0, 100.0));
    }

    #[test]
    fn intersects_respects_t1() {
        let bb = Aabb::from_corners(vec3(-1.0, -2.0, -3.0), vec3(3.0, 2.0, 1.0));
        let ray = RayData::new(vec3(9.0, 9.0, 9.0), vec3(-0.33, -0.33, -0.33));
        assert!(!bb.intersects(&ray, 0.0, 24.0));
    }

    #[test]
    fn intersects_miss_in_one_dimension() {
        // The ray "hits" in the x and z dimensions but not in the y dimension.
        // If one inadvertedly takes the *union* of the slab intersection intervals, as an
        // early version of a certain crate did, this would (incorrectly) be considered a hit.
        // (The numbers are ugly because they're taken from a real ray tracer.)
        let bb = Aabb::from_corners(vec3(-1.5569899, -1.543336, -8.447131),
                                    vec3(1.5569899, 1.5433359, -6.033665));
        let r = RayData::new(vec3(0.0, 0.0, 0.0),
                             vec3(-0.17893936, 0.6952105, -0.6961774));
        assert!(!bb.intersects(&r, 0.0, f32::INFINITY));
    }

    #[test]
    fn centroid() {
        let bb = Aabb::from_corners(vec3(-1.0, 0.0, 5.5), vec3(1.0, 3.0, 10.0));
        assert_eq!(bb.centroid(), vec3(0.0, 1.5, 7.75));
    }

    #[test]
    fn largest_axis() {
        let mut bb = Aabb::from_corners(vec3(0.0, 0.0, 0.0), vec3(1.0, 2.0, 3.0));
        assert_eq!(bb.largest_axis(), 2);
        bb.add_point(vec3(0.0, 4.0, 0.0));
        assert_eq!(bb.largest_axis(), 1);
        bb.add_point(vec3(5.0, 0.0, 0.0));
        bb.add_point(vec3(0.0, 0.0, 5.0));
        assert!(bb.largest_axis() == 0 || bb.largest_axis() == 2);
    }

    #[test]
    #[should_panic]
    fn empty_min() {
        Aabb::empty().min();
    }

    #[test]
    #[should_panic]
    fn empty_max() {
        Aabb::empty().max();
    }

    #[test]
    #[should_panic]
    fn flipped_corners() {
        Aabb::from_corners(vec3(1.0, 2.0, 3.0), vec3(3.0, 2.0, 1.0));
    }

    #[test]
    #[should_panic]
    fn empty_corners() {
        let inf = vec3(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        Aabb::from_corners(inf, -inf);
    }

    #[test]
    #[should_panic]
    fn empty_box_centroid() {
        Aabb::empty().centroid();
    }

    #[test]
    #[should_panic]
    fn empty_surface_area() {
        Aabb::empty().surface_area();
    }
}
