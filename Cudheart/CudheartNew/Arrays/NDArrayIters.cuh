#pragma once

#include <iterator>

namespace CudheartNew {
    class iterator : public std::iterator<std::random_access_iterator_tag, int> {
    public:
        explicit iterator(int* ptr) : ptr_(ptr) {}

        iterator& operator++() {
            ++ptr_;
            return *this;
        }

        iterator operator++(int) {
            iterator tmp(*this); ++ptr_;
            return tmp;
        }

        iterator& operator--() {
            --ptr_;
            return *this;
        }

        iterator operator--(int) {
            iterator tmp(*this);
            --ptr_;
            return tmp;
        }

        bool operator==(const iterator& other) const {
            return ptr_ == other.ptr_;
        }

        bool operator!=(const iterator& other) const {
            return ptr_ != other.ptr_;
        }

        int& operator*() {
            return *ptr_;
        }

    private:
        int* ptr_;
    };

    class const_iterator : public std::iterator<std::random_access_iterator_tag, int> {
    public:
        explicit const_iterator(int const* ptr) : ptr_(ptr) {}

        const_iterator& operator++() {
            ++ptr_;
            return *this;
        }

        const_iterator operator++(int) {
            const_iterator tmp(*this);
            ++ptr_;
            return tmp;
        }

        const_iterator& operator--() {
            --ptr_;
            return *this;
        }

        const_iterator operator--(int) {
            const_iterator tmp(*this);
            --ptr_;
            return tmp;
        }

        bool operator==(const const_iterator& other) const {
            return ptr_ == other.ptr_;
        }

        bool operator!=(const const_iterator& other) const {
            return ptr_ != other.ptr_;
        }

        const int& operator*() {
            return *ptr_;
        }

    private:
        const int* ptr_;
    };
}