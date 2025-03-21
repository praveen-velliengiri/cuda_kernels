#include<cstddef>
#include<iostream>

template<typename T>
T* getmem(size_t n) {
  T* mem = (T*) malloc(sizeof(T) * n);
  memset(mem, 0, sizeof(T) * n);
  return mem;
}

template<typename T>
void fillmem(T* ptr, T val, size_t n) {
  for (int i=0; i<n; i++)
    ptr[i] = val;
}

template<typename T>
void fillseq(T* ptr, T val, size_t n) {
  for (int i=0; i<n; i++, val++)
    ptr[i] = val;
}

template<typename T>
void fillran(T* ptr, size_t n, size_t mod = 1) {
  for (int i=0; i<n; i++)
    ptr[i] = i % mod;
}

template<typename T>
bool check(T* cpu, T* gpu, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (cpu[i] != gpu[i]) {
      std::cout << "\n[ERROR] Mismatch detected at index " << i << ":\n"
                << "  Expected: " << cpu[i] << "\n"
                << "  Received: " << gpu[i] << std::endl;
      return false;
    }
  }
  return true;
}