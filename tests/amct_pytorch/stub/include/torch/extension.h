#ifndef EXTENSION_H
#define EXTENSION_H

#include <vector>
#include <cmath>
#include <string>
#include <stdio.h>
#include <tuple>
#include <iostream>
#include <stdint.h>

static int TORCH_EXTENSION_NAME = 115;

namespace torch {
    const char kInt8 = '8';
    const char kUInt8 = '8';
    const char kCUDA = 'u';
    const char kInt32 = 'k';
    const char kFloat32 = 'f';
    const char kFloat64 = 'd';
    const char kFloat16 = 's';
    const char kBFloat16 = 's';
    const char kFloat8_e4m3fn = '8';
    // ******************************************************************************
    // ************************************ArrayRef**********************************
    // ******************************************************************************
    template<typename T>
    class ArrayRef {
        public:
            std::vector<T> shape_data;
        public:
            ArrayRef(void) {}

            ArrayRef(std::initializer_list<T> init) : shape_data(init) {
                std::cout << "ArrayRef(initializer_list) " << std::endl;
                for (int i=0; i<shape_data.size(); i++) {
                }
            }

            void append(std::initializer_list<T> list) {
                shape_data.insert(shape_data.end(), list.begin(), list.end());
            }
            /// equals - Check for element-wise equality.
            bool equals(ArrayRef RHS) {
                return shape_data == RHS.shape_data;
            }

            std::vector<T> vec() {
                return shape_data;
            }

            void clear() {
                shape_data.clear();
            }

            T* data() {
                return &shape_data[0];
            }

            /// size - Get the array size.
            size_t size() {
                return shape_data.size();
            }
            T& at(size_t Index) {
                return shape_data[Index];
            }

            T& operator[](size_t Index) {
                return shape_data[Index];
            }

            void set(size_t Index, T value){
                shape_data[Index] = value;
            }

    }; // class ArrayRef

    template<typename T>
    static bool operator==(ArrayRef<T> a1, ArrayRef<T> a2) {
      return a1.equals(a2);
    }

    template<typename T>
    static bool operator!=(ArrayRef<T> a1, ArrayRef<T> a2) {
      return !a1.equals(a2);
    }
    // ******************************************************************************
    // ************************************IntArrayRef*******************************
    // ******************************************************************************
    class IntArrayRef {
        public:
            std::vector<int64_t> shape_data;
        public:
            IntArrayRef(void) {}

            IntArrayRef(std::initializer_list<int64_t> init) : shape_data(init) {
                std::cout << "IntArrayRef(initializer_list) " << std::endl;
                for (int i=0; i<shape_data.size(); i++) {
                }
            }

            void append(std::initializer_list<int64_t> list) {
                shape_data.insert(shape_data.end(), list.begin(), list.end());
            }
            /// equals - Check for element-wise equality.
            bool equals(IntArrayRef RHS) {
                return shape_data == RHS.shape_data;
            }

            std::vector<int64_t> vec() {
                return shape_data;
            }

            void clear() {
                shape_data.clear();
            }

            void show() {
                printf("IntArrayRef::show_data\n");
                for (int i=0; i<shape_data.size(); i++) {
                    printf("%ld ", shape_data[i]);
                }
                printf("\n");
            }

            int64_t* data() {
                return &shape_data[0];
            }

            /// size - Get the array size.
            size_t size() {
                return shape_data.size();
            }
            int64_t& at(size_t Index) {
                return shape_data[Index];
            }

            int64_t& operator[](size_t Index) {
                return shape_data[Index];
            }

            void set(size_t Index, int64_t value){
                shape_data[Index] = value;
            }

            int64_t length() {
                int64_t length = 1;
                for (int i=0; i<shape_data.size(); i++) {
                    length = length * shape_data[i];
                }
                return length;
            }

            void reshape(std::initializer_list<int64_t> list) {
                int64_t len = length();
                shape_data.clear();
                shape_data.insert(shape_data.end(), list.begin(), list.end());

                int64_t len_new = length();
                int64_t undefine_shape =  len / abs(len_new);
                for (size_t i=0; i<shape_data.size(); i++){
                    if (shape_data[i] == -1){
                        shape_data[i] = undefine_shape;
                        break;
                    }
                }
            }
    }; // class IntArrayRef


    static bool operator==(IntArrayRef a1, IntArrayRef a2) {
      return a1.equals(a2);
    }

    static bool operator!=(IntArrayRef a1, IntArrayRef a2) {
      return !a1.equals(a2);
    }

    // ******************************************************************************
    // ************************************Tensor************************************
    // ******************************************************************************
    class Tensor
    {
        public:
            std::vector<float> tensor_data;
            float* tensor_data_ptr=nullptr;
            IntArrayRef sizes_data{};
            bool device_cuda;

            // flag: float, double, int, not support.
            std::vector<bool> type_flag{false, false, false, false, false, false};

            Tensor(Tensor* const input){
                device_cuda = input->device_cuda;
                for (int i = 0; i<input->tensor_data.size(); i++) {
                    float val = static_cast<float>(input->tensor_data[i]);
                    tensor_data.push_back(val);
                }

                for (size_t i = 0; i < input->sizes_data.size(); i++) {
                    sizes_data.append({input->sizes_data[i]});
                }
                tensor_data_ptr = &tensor_data[0];

                for(int idx = 0; idx < input->type_flag.size(); ++idx)
                {
                    if(input->type_flag[idx])
                    {
                        type_flag[idx] = true;
                        break;
                    }
                }
            }

            template<typename T>
            Tensor(std::vector<T> input){
                device_cuda = true;
                for (int i = 0; i< input.size(); i++) {
                    tensor_data.push_back(static_cast<float>(input[i]));
                }
                sizes_data.append({static_cast<int64_t>(input.size())});
                tensor_data_ptr = &tensor_data[0];
                //sizes_data.show();

                if(typeid(T) == typeid(float))
                    type_flag[0] = true;
                else if(typeid(T) == typeid(double))
                    type_flag[1] = true;
                else if(typeid(T) == typeid(int))
                    type_flag[2] = true;
                else if(typeid(T) == typeid(uint8_t))
                    type_flag[3] = true;
                else if(typeid(T) == typeid(uint16_t))
                    type_flag[4] = true;
                else
                    type_flag[5] = true;
            }
            //template<typename T>
            Tensor(float input){
                device_cuda = true;
                tensor_data.push_back(static_cast<float>(input));
                sizes_data.append({0});
                tensor_data_ptr = &tensor_data[0];
            }

            Tensor(void){}

            /// Returns if a `Tensor` has CUDA backend.
            bool is_cuda(){
                return device_cuda;
            }

            Tensor cpu() {
                this->device_cuda = false;
                return this;
            }

            Tensor cuda() {
                this->device_cuda  = true;
                return this;
            }

            char device() {
                if (device_cuda) {
                    //return Device('u');
                    return 'u';
                } else {
                    //return Device('c');
                    return 'u';
                }
            }
            template <typename T=float>
            T * data_ptr() const {
                static std::vector<T> tensor_data_new;
                tensor_data_new.clear();
                for (int i = 0; i < tensor_data.size(); i++) {
                    tensor_data_new.push_back(static_cast<T>(tensor_data[i]));
                }
                return tensor_data_new.data();
            }

            template<typename T>
            T * data() const {
                return data_ptr<T>();
            }

            Tensor operator[](size_t Index) {
                return this;
            }

            Tensor operator-() {
                return this;
            }

            template<typename T>
            T item() {
                return static_cast<T>(tensor_data[0]);
            }

            char dtype(char dtype){
                return dtype;
            }

            char dtype()
            {

                if(type_flag[0])
                    return 'f'; // float dtype.
                else if(type_flag[1])
                    return 'd'; // double dtype.
                else if(type_flag[2])
                    return 'k'; // int dtype.
                else if (type_flag[3])
                    return 's'; // float16 dtype
                else
                    return 'n'; // not support dtype.
            }

            int64_t numel() const {
                return tensor_data.size();
            }

            Tensor contiguous() {
                // Tensor ret = Tensor(this->tensor_data[0]);
                return this;
            }

            Tensor detach() {
                return this;
            }

            Tensor from_blob(int* data, std::initializer_list<long> shape, char dtype){
                return this;
            }

            IntArrayRef sizes() const {
                return sizes_data;
            }

            int64_t size(int dim)
            {
                return sizes_data.shape_data[dim];
            }

            size_t dim() {
                return sizes_data.size();
            }

            Tensor to(const char type) {
                return this;
            }

            Tensor to(const std::string type) {
                return this;
            }

            Tensor clone() {
                return this;
            }

            Tensor reshape(std::initializer_list<long> new_list){
                //printf("Tensor::reshape(initializer_list&)\n");
                sizes_data.reshape(new_list);
                return this;
            }

            Tensor reshape(IntArrayRef& new_size){
                //printf("Tensor::reshape(IntArrayRef&)\n");
                sizes_data = new_size;
                return this;
            }

            Tensor reshape(std::vector<int64_t>& new_size){
                return this;
            }


            Tensor transpose(int dim0, int dim1){
                int64_t shape0 = sizes_data[dim0];
                int64_t shape1 = sizes_data[dim1];
                sizes_data.set(dim0, shape1);
                sizes_data.set(dim1, shape0);
                return this;
            }

            void index_put_(std::initializer_list<long> location, float value) {
                return;
            }

            void show_data() {
                printf("Tensor::show_data\n");
                for(int i=0; i<tensor_data.size(); i++) {
                    printf("%f ", tensor_data[i]);
                }
                printf("\n");
            }

            void clear() {
                //printf("Tensor::clear\n");
                tensor_data.clear();
                sizes_data.clear();
                //sizes_data.show();
                this->show_data();
                tensor_data_ptr=nullptr;
            }
            // Tensor operators
            Tensor clamp(float min, float max){
                return this;
            }

            Tensor clamp_(float min, float max){
                return this;
            }

            Tensor clamp_max_(float max){
                return this;
            }

            Tensor clamp_min_(float min){
                return this;
            }

            template<typename T>
            Tensor add(T right){
                return this;
            }

            template<typename T>
            Tensor add_(T right){
                return this;
            }

            template<typename T>
            Tensor sub(T right){
                return this;
            }

            template<typename T>
            Tensor sub_(T right){
                return this;
            }

            template<typename T>
            Tensor div(T right){
                return this;
            }

            template<typename T>
            Tensor div_(T right){
                return this;
            }

            template<typename T>
            Tensor mul(T right){
                return this;
            }

            template<typename T>
            Tensor mul_(T right){
                return this;
            }

            template<typename T>
            Tensor pow(T power){
                return this;
            }

            Tensor sum(){
                return this;
            }

            Tensor round(){
                return this;
            }

            Tensor round_(){
                return this;
            }

            Tensor neg(){
                return this;
            }

            Tensor neg_(){
                return this;
            }

            Tensor argmin(int64_t dim) {
                return this;
            }

            Tensor floor(){
                return this;
            }

            Tensor floor_(){
                return this;
            }

            Tensor norm(int p, int dim){
                return this;
            }

            Tensor norm(int dim){
                return this;
            }
            // compare
            bool is_same_size(Tensor other) {
                return sizes_data == other.sizes_data;
            }

            Tensor min(Tensor other) {
                return Tensor(std::vector<float>{-1.0, -1.0});
            }

            Tensor max(Tensor other) {
                return Tensor(std::vector<float>{1.0, 1.0});;
            }

            Tensor reciprocal(){
                return this;
            }

            Tensor reciprocal_(){
                return this;
            }

            Tensor sqrt(){
                return this;
            }

            Tensor sqrt_(){
                return this;
            }

            template<typename T>
            Tensor sum(T dim){
                return this;
            }

            Tensor abs(){
                return this;
            }

            Tensor scatter_(int axis, Tensor indices, float data){
                return this;
            }

            std::tuple<Tensor, Tensor> topk(int n, int axis, bool flag){
                std::tuple<Tensor, Tensor> tup;
                return tup;
            }
    };

    static bool operator==(Tensor a1, Tensor a2) {
      return true;
    }

    static bool operator!=(Tensor a1, Tensor a2) {
      return true;
    }

    static Tensor tensor(int data){
        Tensor ret= Tensor(data);
        return ret;
    }

    static Tensor scalar_tensor(int data){
        Tensor ret= Tensor(data);
        return ret;
    }

    static Tensor zeros_like(Tensor other) {
        Tensor new_tensor(std::vector<float>{0.0});
        return new_tensor;
    }

    static Tensor zeros(Tensor other, const char type) {
        Tensor new_tensor(std::vector<float>{0.0});
        return new_tensor;
    }

    static Tensor zeros(std::initializer_list<long> shape) {
        Tensor new_tensor(std::vector<float>{0.0});
        return new_tensor;
    }

    static Tensor zeros(std::initializer_list<long> shape,  const char type) {
        Tensor new_tensor(std::vector<float>{0.0});
        return new_tensor;
    }

    static Tensor zeros(IntArrayRef shape, const char dtype) {
        Tensor new_tensor(std::vector<float>{0.0});
        return new_tensor;
    }

    static Tensor ones(std::initializer_list<long> shape, const char dtype) {
        Tensor new_tensor(std::vector<float>{1.0});
        return new_tensor;
    }

    static Tensor ones(IntArrayRef shape, const char dtype) {
        Tensor new_tensor(std::vector<float>{1.0});
        return new_tensor;
    }

    static std::tuple<Tensor, Tensor> max(Tensor other, int64_t dim, bool keepdim=false) {
        Tensor index(std::vector<float>{0.0});
        std::tuple<Tensor, Tensor> result(index, other);
        return result;
    }

    static Tensor max(Tensor tensor, Tensor other) {
        return tensor;
    }

    static std::tuple<Tensor, Tensor>  min(Tensor other, int64_t dim, bool keepdim=false){
        Tensor index(std::vector<float>{0.0});
        std::tuple<Tensor, Tensor> result(index, other);
        return result;
    }

    static Tensor cat(std::initializer_list<Tensor> tensorlist, int64_t dim){
        Tensor result = *tensorlist.begin();
        return result;
    }

    template<typename T>
    static Tensor from_blob(T* data, std::initializer_list<long> shape, char dtype){
        return Tensor(std::vector<float>{0.0});
    }

    template<typename T>
    static Tensor from_blob(T* data, IntArrayRef shape, char dtype){
        return Tensor(std::vector<float>{0.0});
    }

    static torch::Tensor device(char device){
        return torch::Tensor(std::vector<float>{1.0});
    }

    static Tensor abs(Tensor data) {
        return data;
    }

    template<typename T>
    static Tensor operator>(Tensor a1, T a2) {
        return a1;
    }

    template<typename T>
    static Tensor operator<(Tensor a1, T a2) {
        return a1;
    }

    template<typename T>
    static Tensor operator/(Tensor a1, T a2) {
        return a1;
    }

    template<typename T>
    static Tensor operator-(Tensor a1, T a2) {
        return a1;
    }

    template<typename T>
    static Tensor operator+(Tensor a1, T a2) {
        return a1;
    }

    template<typename T>
    static Tensor operator*(Tensor a1, T a2) {
        return a1;
    }

    static Tensor ones_like(Tensor data) {
        return data;
    }

    static std::vector<Tensor> split(Tensor tensor, int a, int b){
        std::vector<Tensor> vec;
        vec.push_back(tensor);
        return vec;
    }
};

namespace c10 {
    const char kCPU = 'c';
    const char kCUDA = 'u';
    const char kFloat = 'f';
    const char kFloat32 = 'f';
};

namespace at {
    static torch::Tensor where(torch::Tensor condition, torch::Tensor t, torch::Tensor f) {
        return condition;
    }
    class IntArrayRef {
        public:
            std::vector<int64_t> shape_data;
        public:
            IntArrayRef(void) {}

            IntArrayRef(std::initializer_list<int64_t> init) : shape_data(init) {
                std::cout << "IntArrayRef(initializer_list) " << std::endl;
                for (int i=0; i<shape_data.size(); i++) {
                }
            }

            void append(std::initializer_list<int64_t> list) {
                shape_data.insert(shape_data.end(), list.begin(), list.end());
            }
            /// equals - Check for element-wise equality.
            bool equals(IntArrayRef RHS) {
                return shape_data == RHS.shape_data;
            }

            std::vector<int64_t> vec() {
                return shape_data;
            }

            void clear() {
                shape_data.clear();
            }

            void show() {
                printf("IntArrayRef::show_data\n");
                for (int i=0; i<shape_data.size(); i++) {
                    printf("%ld ", shape_data[i]);
                }
                printf("\n");
            }

            int64_t* data() {
                return &shape_data[0];
            }

            /// size - Get the array size.
            size_t size() {
                return shape_data.size();
            }
            int64_t& at(size_t Index) {
                return shape_data[Index];
            }

            int64_t& operator[](size_t Index) {
                return shape_data[Index];
            }

            void set(size_t Index, int64_t value){
                shape_data[Index] = value;
            }

            int64_t length() {
                int64_t length = 1;
                for (int i=0; i<shape_data.size(); i++) {
                    length = length * shape_data[i];
                }
                return length;
            }

            void reshape(std::initializer_list<int64_t> list) {
                int64_t len = length();
                shape_data.clear();
                shape_data.insert(shape_data.end(), list.begin(), list.end());

                int64_t len_new = length();
                int64_t undefine_shape =  len / abs(len_new);
                for (size_t i=0; i<shape_data.size(); i++){
                    if (shape_data[i] == -1){
                        shape_data[i] = undefine_shape;
                        break;
                    }
                }
            }
    }; // class IntArrayRef

    const char kFloat = 'f';
}

static torch::Tensor device(char device){
    return torch::Tensor(std::vector<float>{1.0});
}

#define PYBIND11_MODULE(name, variable) \
        static void fake(FakeClass &variable)


class FakeClass
{
public:
    template <typename Func, typename... Extra>
    FakeClass &def(const char *name_, Func &&f, const Extra& ... extra) {
        return *this;
    }

    FakeClass &def(const char *name_) {
        return *this;
    };

    template <typename T>
    FakeClass &def_readwrite(const char *name_, T param) {
        return *this;
    };
};

namespace py {
    template <typename... Extra>
    FakeClass class_(FakeClass &variable, const char *name){
        return variable;
    }

    template <typename... Extra>
    const char* init() {
        return "init";
    };
}

#endif /* EXTENSION_H */
