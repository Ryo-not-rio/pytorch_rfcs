# [Vector length agnostic SVE class]

**Authors:**
* @Ryo-not-rio


## **Summary**
A vector length agnostic implementation of the `Vectorized` class for SVE.

## **Motivation**
PyTorch contains a `Vectorized` class that wraps different SIMD intrinsics for different architectures by having a vector as a class attribute, and the class methods acting as the wrapper around the intrinsics.  This works well for non-scalable vectors but poses an issue for SVE due to the inability to store them as class attributes. The current workaround for this is to use the compiler flag `-msve-vector-bits=\<bits\>`, however this is not ideal as this would 1. require separate `Vectorized` classes for different vector lengths and 2. does not allow for runtime detection of the actual vector length. We currently only have an implementation of the `Vectorized` class for 256-bit SVE machines but as we think about adding support for different vector length, we need to consider how to avoid code duplication as raised by @malfet [here](https://github.com/pytorch/pytorch/pull/138388#issuecomment-2635612409). This RFC aims to solve the issue but creating a `Vectorized` class that detects the vector length at runtime as SVE is intended to be used, allowing us to support different vector lengths without writing any duplicate code.

## **Proposed Implementation**
The basic premise of our proposal is to store not the SVE vector but an array in our `Vectorized` class which we will load from and store to with each operation. A minimal version is shown at the end.

Now this introduces quite an obvious overhead of an additional load and store operation with each op. However, the compiler is able to optimize these out with the following conditions:

1. The -O3 flag is set
2. The `svptrue_b32()` predicate is used
3. You are storing to and then loading from the same pointer

Ensuring these conditions are met and by inlining the functions, we can rely on the compiler to optimize the duplicate load and stores, ensuring we do not introduce any regressions.

### The size problem
We face a challenge with this implementation due to the constraint of the size() function being constexpr. The size() function which returns the number of elements in the `Vectorized` class cannot be constexpr in our implmentation due to SVE vector lengths being unknown at compile time. We propose we change this to be const instead of constexpr. Currently, size() is used to initialize std::arrays and to instantiate templated functions. These will need to be replaced with c arrays and the template parameters made into function arguments. The full list of changes that need to occur can be seen [here](https://github.com/pytorch/pytorch/commit/fa05c1de3340215da5dc0a32612e75e2816fc143).

```
class Vectorized<float> {
	float values[64]; // Maximum number of elements supported by any SVE machine

    static inline const size_type size() {
        return svcntw();
    }
    static inline Vectorized<float> loadu(const float * vs) {
        Vectorized<float> v;
        svfloat32_t vec = svld1_f32(svptrue_b32(), static_cast<const float *>(vs));
        svst1_f32(svptrue_b32(), v.values, vec);
        return v;
    }

    inline void store(void* ptr) const {
        svfloat32_t vec = svld1_f32(svptrue_b32(), values);
        svst1_f32(svptrue_b32(), static_cast<float *>(ptr), vec);
    }

    inline Vectorized<float> abs() const {
		svfloat32_t v = svld1_f32(svptrue_b32(), values);
    	v = svabs_f32_x(svptrue_b32(), *this);
     	svst1_f32(svptrue_b32(), values, v);
		return *this;
  	}
}
```

## **Metrics **
- Reduction of code duplication
- Speedup of PyTorch on SVE machines with non-256 bit vectors
  - Softmax sped up by 2.73x on Neoverse V2
  - X * sigmoid * softmax sped up by 1.65x on Neoverse V2
- No speed or accuracy regression on 256-bit vectors


## **Drawbacks**
### Implementation cost
This is a large change which requires an overhaul of all of the current SVE `Vectorized` as well as any code that expects the size() function to be constexpr. The first cost can be mitigated by updating the `Vectorized` classes one by one.

### Sideffects from non-constexpr size()
By changing the size() to non-constexpr, we will be changing a large part of the codebase which may cause regressions. These will need to be benchmarked thoroughly and if we choose to accept any regressions, they will need to be limited to aarch64 architectures.

### Memory footprint increase
By storing an array with the size "max SVE vector length (2048 bits being the maximum possible and 512 bits being the longest hardware available)", the memory footprint is increased by `2048 bits x number of existing Vectorized classes`. Since `Vectorized` classes are created and destoryed in loops with only a few instances existing simultaneously, we expect this effect to be minimal, but we should benchmark this using actual models. We could also limit this effect by using the maximum vector size currently available on hardware with scope to change this if necessary.

## **Benchmarking plan**
To mitigate the risk from changing the size() from constexpr to const, we propose the following order of patches to PyTorch:

1. Make individual pull requests for each function affected by this change
2. Bench mark each patch thoroughly both on aarch64 and x86 for regressions
3. Once all affected functions are merged, switch the Vectorized class to the VLA implementation
4. Benchmark the VLA Vectorized class

## **Alternatives**
To keep the size() function constexpr, we considered setting the size of the `Vectorized` class to be the maximum possible SVE vector length and loading multiple vectors as necessary. However, this poses the following problems:

1. Increases the tail loop size unecessarily for machines with smaller vector lengths.
2. Compiler can no longer optimize out duplicate loads and stores as multiple vectors need to be handled consecutively.
3. Detection of number of vectors needed introduces an extra overhead

Due to these issues combined, especially 2., this alternative introduces a ~30% overhead compared to the current implementation.

## **Unresolved questions**
* How much of the existing code has to change due to changing the size() from constexpr to const
* What if any performance regression will we see due to changing size() from constexpr to const?
* Will we have to potentially have to write code specific to SVE due to the size() change?


## Resolution
TBD

### Level of Support
Choose one of the following:
* 1: Overwhelming positive feedback.
* 2: Positive feedback.
* 3: Majority Acceptance, with conflicting Feedback.
* 4: Acceptance, with Little Feedback.
* 5: Unclear Resolution.
* 6: RFC Rejected.
* 7: RFC Rejected, with Conflicting Feedback.


#### Additional Context
[Working proof of concept code](https://github.com/Ryo-not-rio/pytorch/commit/b2e5c66017fb48230d1ea2493b8548ad76d88fcf)


### Next Steps
TBD


#### Tracking issue
https://github.com/pytorch/pytorch/issues/153471


#### Exceptions
TBD