/* ------------------------------------------------------------------------- *
 *                                SPHinXsys                                  *
 * ------------------------------------------------------------------------- *
 * SPHinXsys (pronunciation: s'finksis) is an acronym from Smoothed Particle *
 * Hydrodynamics for industrial compleX systems. It provides C++ APIs for    *
 * physical accurate simulation and aims to model coupled industrial dynamic *
 * systems including fluid, solid, multi-body dynamics and beyond with SPH   *
 * (smoothed particle hydrodynamics), a meshless computational method using  *
 * particle discretization.                                                  *
 *                                                                           *
 * SPHinXsys is partially funded by German Research Foundation               *
 * (Deutsche Forschungsgemeinschaft) DFG HU1527/6-1, HU1527/10-1,            *
 *  HU1527/12-1 and HU1527/12-4.                                             *
 *                                                                           *
 * Portions copyright (c) 2017-2023 Technical University of Munich and       *
 * the authors' affiliations.                                                *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may   *
 * not use this file except in compliance with the License. You may obtain a *
 * copy of the License at http://www.apache.org/licenses/LICENSE-2.0.        *
 *                                                                           *
 * ------------------------------------------------------------------------- */
/**
 * @file 	vector_functions.h
 * @brief 	Basic functions for vector type data.
 * @author	Chi ZHang and Xiangyu Hu
 */
#ifndef SMALL_VECTORS_H
#define SMALL_VECTORS_H

#include "base_data_type.h"
#include "execution_event.h"
#include "execution_queue.h"

namespace SPH
{
Vec2d FirstAxisVector(const Vec2d &zero_vector);
Vec3d FirstAxisVector(const Vec3d &zero_vector);

Vec3d upgradeToVec3d(const Real &input);
Vec3d upgradeToVec3d(const Vec2d &input);
Vec3d upgradeToVec3d(const Vec3d &input);
Mat3d upgradeToMat3d(const Mat2d &input);
Mat3d upgradeToMat3d(const Mat3d &input);

void degradeToVecd(const Vec3d &input, Vec2d &output);
inline void degradeToVecd(const Vec3d &input, Vec3d &output) { output = input; };
void degradeToMatd(const Mat3d &input, Mat2d &output);
inline void degradeToMatd(const Mat3d &input, Mat3d &output) { output = input; };

Mat2d getInverse(const Mat2d &A);
Mat3d getInverse(const Mat3d &A);
Mat2d getAverageValue(const Mat2d &A, const Mat2d &B);
Mat3d getAverageValue(const Mat3d &A, const Mat3d &B);
Mat2d inverseCholeskyDecomposition(const Mat2d &A);
Mat3d inverseCholeskyDecomposition(const Mat3d &A);
Mat2d getDiagonal(const Mat2d &A);
Mat3d getDiagonal(const Mat3d &A);

/** Real dot product between two matrices, resulting in a scalar value (sum of products of element-wise) */
Real CalculateBiDotProduct(Mat2d Matrix1, Mat2d Matrix2); // calculate Real dot
Real CalculateBiDotProduct(Mat3d Matrix1, Mat3d Matrix2); // calculate Real dot

/** get transformation matrix. */
Mat2d getTransformationMatrix(const Vec2d &direction_of_y);
Mat3d getTransformationMatrix(const Vec3d &direction_of_z);

/** get angle between two vectors. */
Real getCosineOfAngleBetweenTwoVectors(const Vec2d &vector_1, const Vec2d &vector_2);
Real getCosineOfAngleBetweenTwoVectors(const Vec3d &vector_1, const Vec3d &vector_2);

/** get orthogonal projection of a vector. */
Vec2d getVectorProjectionOfVector(const Vec2d &vector_1, const Vec2d &vector_2);
Vec3d getVectorProjectionOfVector(const Vec3d &vector_1, const Vec3d &vector_2);

/** von Mises stress from stress matrix */
Real getVonMisesStressFromMatrix(const Mat2d &sigma);
Real getVonMisesStressFromMatrix(const Mat3d &sigma);

/** principal strain or stress from strain or stress matrix */
Vec2d getPrincipalValuesFromMatrix(const Mat2d &A);
Vec3d getPrincipalValuesFromMatrix(const Mat3d &A);

/** get transformation matrix. */
Real getCrossProduct(const Vec2d &vector_1, const Vec2d &vector_2);
Vec3d getCrossProduct(const Vec3d &vector_1, const Vec3d &vector_2);

/** convert host Vecd to device Vecd */
inline DeviceVec2d hostToDeviceVecd(const Vec2d& host) { return {host[0], host[1]}; }
inline DeviceVec3d hostToDeviceVecd(const Vec3d& host) { return {host[0], host[1], host[2]}; }
inline DeviceArray2i hostToDeviceArrayi(const Array2i& host) { return {host[0], host[1]}; }
inline DeviceArray3i hostToDeviceArrayi(const Array3i& host) { return {host[0], host[1], host[2]}; }

/** convert device Vecd to host Vecd */
inline Vec2d deviceToHostVecd(const DeviceVec2d& device) { return {device[0], device[1]}; }
inline Vec3d deviceToHostVecd(const DeviceVec3d& device) { return {device[0], device[1], device[2]}; }

/** Initialize Vecd of zeros for host or device */
template<class V> inline V VecdZero();
template<> inline DeviceVec2d VecdZero() { return DeviceVec2d{0}; }
template<> inline DeviceVec3d VecdZero() { return DeviceVec3d{0}; }
template<> inline Vec2d VecdZero() { return Vec2d::Zero(); }
template<> inline Vec3d VecdZero() { return Vec3d::Zero(); }

/* specialization for specific vector operations */
template<class RealType, int Dimension>
inline RealType VecdDot(const sycl::vec<RealType,Dimension>& v1, const sycl::vec<RealType,Dimension>& v2) {
    return sycl::dot(v1, v2);
}
template<class RealType, int Dimension>
inline RealType VecdDot(const Eigen::Matrix<RealType,Dimension,1>& v1, const Eigen::Matrix<RealType,Dimension,1>& v2) {
    return v1.dot(v2);
}
template<class RealType, int Dimension>
inline RealType VecdNorm(const sycl::vec<RealType,Dimension>& vec) {
    return sycl::length(vec);
}
template<class RealType, int Dimension>
inline RealType VecdNorm(const Eigen::Matrix<RealType,Dimension,1>& vec) {
    return vec.norm();
}
template<class RealType, int Dimension>
inline RealType VecdSquareNorm(const sycl::vec<RealType,Dimension>& vec) {
    return sycl::dot(vec, vec);
}
template<class RealType, int Dimension>
inline RealType VecdSquareNorm(const Eigen::Matrix<RealType,Dimension,1>& vec) {
    return vec.squaredNorm();
}

template<class VecType, std::size_t ...Index>
inline auto VecdFoldingProd_impl(const VecType& vec, std::index_sequence<Index...>) {
    return ( vec[Index] * ... );
}
template<class Type, int Dimension>
inline Type VecdFoldProduct(const sycl::vec<Type,Dimension>& vec) {
    return VecdFoldingProd_impl(vec, std::make_index_sequence<Dimension>());
}

/* SYCL memory transfer utilities */
template<class T>
inline T* allocateDeviceData(std::size_t size) {
    return sycl::malloc_device<T>(size, execution::executionQueue.getQueue());
}

template<class T>
inline T* allocateSharedData(std::size_t size) {
    return sycl::malloc_shared<T>(size, execution::executionQueue.getQueue());
}

template<class T>
inline void freeDeviceData(T* device_mem) {
    sycl::free(device_mem, execution::executionQueue.getQueue());
}

template<class T>
inline execution::ExecutionEvent copyDataToDevice(const T* host, T* device, std::size_t size) {
    return execution::executionQueue.getQueue().memcpy(device, host, size*sizeof(T));
}

template<class HostType, class DeviceType, class TransformFunc>
execution::ExecutionEvent transformAndCopyDataToDevice(const HostType* host, DeviceType* device, std::size_t size, TransformFunc&& transformation) {
    auto* transformed_host_copy = new DeviceType[size];
    for(size_t i = 0; i < size; ++i)
        transformed_host_copy[i] = transformation(host[i]);
    return std::move(copyDataToDevice(transformed_host_copy, device, size).then([=](){ delete[] transformed_host_copy; }));
}

inline execution::ExecutionEvent copyDataToDevice(const Vec2d* host, DeviceVec2d* device, std::size_t size) {
    return transformAndCopyDataToDevice(host, device, size, [](auto vec) { return hostToDeviceVecd(vec); });
}

inline execution::ExecutionEvent copyDataToDevice(const Vec3d* host, DeviceVec3d* device, std::size_t size) {
    return transformAndCopyDataToDevice(host, device, size, [](auto vec) { return hostToDeviceVecd(vec); });
}

inline execution::ExecutionEvent copyDataToDevice(const Real* host, DeviceReal* device, std::size_t size) {
    return transformAndCopyDataToDevice(host, device, size, [](Real val) { return static_cast<DeviceReal>(val); });
}

template<class T>
inline execution::ExecutionEvent copyDataToDevice(const T& value, T* device, std::size_t size) {
    return execution::executionQueue.getQueue().fill(device, value, size);
}

template<class T>
inline execution::ExecutionEvent copyDataFromDevice(T* host, const T* device, std::size_t size) {
    return execution::executionQueue.getQueue().memcpy(host, device, size*sizeof(T));
}

template<class HostType, class DeviceType, class TransformFunc>
execution::ExecutionEvent transformAndCopyDataFromDevice(HostType* host, const DeviceType* device, std::size_t size, TransformFunc&& transformation) {
    auto* device_copy = new DeviceType[size];
    return std::move(copyDataFromDevice(device_copy, device, size).then([=](){
                                                                  for(size_t i = 0; i < size; ++i)
                                                                      host[i] = transformation(device_copy[i]);
                                                                  delete[] device_copy;
                                                              }));
}


inline execution::ExecutionEvent copyDataFromDevice(Vec2d* host, const DeviceVec2d* device, std::size_t size) {
    return transformAndCopyDataFromDevice(host, device, size, [](auto vec) { return deviceToHostVecd(vec); });
}

inline execution::ExecutionEvent copyDataFromDevice(Vec3d* host, const DeviceVec3d* device, std::size_t size) {
    return transformAndCopyDataFromDevice(host, device, size, [](auto vec) { return deviceToHostVecd(vec); });
}

inline execution::ExecutionEvent copyDataFromDevice(Real* host, const DeviceReal* device, std::size_t size) {
    return transformAndCopyDataFromDevice(host, device, size, [](DeviceReal val) { return static_cast<Real>(val); });
}

/** Bounding box for system, body, body part and shape, first: lower bound, second: upper bound. */
template <typename VecType>
class BaseBoundingBox
{
  public:
    VecType first_, second_;
    int dimension_;

    BaseBoundingBox() : first_(VecType::Zero()), second_(VecType::Zero()), dimension_(VecType::Zero().size()){};
    BaseBoundingBox(const VecType &lower_bound, const VecType &upper_bound)
        : first_(lower_bound), second_(upper_bound), dimension_(lower_bound.size()){};
    /** Check the bounding box contain. */
    bool checkContain(const VecType &point)
    {
        bool is_contain = true;
        for (int i = 0; i < dimension_; ++i)
        {
            if (point[i] < first_[i] || point[i] > second_[i])
            {
                is_contain = false;
                break;
            }
        }
        return is_contain;
    };
};
/** Operator define. */
template <class T>
bool operator==(const BaseBoundingBox<T> &bb1, const BaseBoundingBox<T> &bb2)
{
    return bb1.first_ == bb2.first_ && bb1.second_ == bb2.second_ ? true : false;
};
/** Intersection fo bounding box.*/
template <class BoundingBoxType>
BoundingBoxType getIntersectionOfBoundingBoxes(const BoundingBoxType &bb1, const BoundingBoxType &bb2)
{
    /** Check that the inputs are correct. */
    int dimension = bb1.dimension_;
    /** Get the Bounding Box of the intersection of the two meshes. */
    BoundingBoxType bb(bb1);
    /** #1 check that there is overlap, if not, exception. */
    for (int i = 0; i < dimension; ++i)
        if (bb2.first_[i] > bb1.second_[i] || bb2.second_[i] < bb1.first_[i])
            std::runtime_error("getIntersectionOfBoundingBoxes: no overlap!");
    /** #2 otherwise modify the first one to get the intersection. */
    for (int i = 0; i < dimension; ++i)
    {
        /** If the lower limit is inside change the lower limit. */
        if (bb1.first_[i] < bb2.first_[i] && bb2.first_[i] < bb1.second_[i])
            bb.first_[i] = bb2.first_[i];
        /**  If the upper limit is inside, change the upper limit. */
        if (bb1.second_[i] > bb2.second_[i] && bb2.second_[i] > bb1.first_[i])
            bb.second_[i] = bb2.second_[i];
    }
    return bb;
}

/** obtain minimum dimension of a bounding box */
template <class BoundingBoxType>
Real MinimumDimension(const BoundingBoxType &bbox)
{
    return (bbox.second_ - bbox.first_).cwiseAbs().minCoeff();
};
} // namespace SPH
#endif // SMALL_VECTORS_H
