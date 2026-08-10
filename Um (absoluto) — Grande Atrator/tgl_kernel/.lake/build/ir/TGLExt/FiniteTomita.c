// Lean compiler output
// Module: TGLExt.FiniteTomita
// Imports: public import Init public meta import Init public import TGLExt.LeftRight
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
extern lean_object* lp_mathlib_Complex_instAddCommMonoid;
extern lean_object* lp_mathlib_Complex_instMul;
lean_object* lp_mathlib_dotProduct___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_trace___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_tgl__kernel_TGLExt_gibbs___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_tgl__kernel_TGLExt_gibbs___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_tgl__kernel_TGLExt_gibbs___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_tgl__kernel_TGLExt_gibbs___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_tgl__kernel_TGLExt_gibbs___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_tgl__kernel_TGLExt_gibbs(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_tgl__kernel_TGLExt_gibbs___redArg___lam__0(lean_object* v_a_1_, lean_object* v___y_2_, lean_object* v_j_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_a_1_, v_j_3_, v___y_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_tgl__kernel_TGLExt_gibbs___redArg___lam__1(lean_object* v_00_u03c1_5_, lean_object* v___y_6_, lean_object* v_j_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lean_apply_2(v_00_u03c1_5_, v___y_6_, v_j_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_tgl__kernel_TGLExt_gibbs___redArg___lam__2(lean_object* v_a_9_, lean_object* v_00_u03c1_10_, lean_object* v_inst_11_, lean_object* v___x_12_, lean_object* v___x_13_, lean_object* v___y_14_, lean_object* v___y_15_){
_start:
{
lean_object* v___f_16_; lean_object* v___f_17_; lean_object* v___x_18_; 
v___f_16_ = lean_alloc_closure((void*)(lp_tgl__kernel_TGLExt_gibbs___redArg___lam__0), 3, 2);
lean_closure_set(v___f_16_, 0, v_a_9_);
lean_closure_set(v___f_16_, 1, v___y_15_);
v___f_17_ = lean_alloc_closure((void*)(lp_tgl__kernel_TGLExt_gibbs___redArg___lam__1), 3, 2);
lean_closure_set(v___f_17_, 0, v_00_u03c1_10_);
lean_closure_set(v___f_17_, 1, v___y_14_);
v___x_18_ = lp_mathlib_dotProduct___redArg(v_inst_11_, v___x_12_, v___x_13_, v___f_17_, v___f_16_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_tgl__kernel_TGLExt_gibbs___redArg___lam__2___boxed(lean_object* v_a_19_, lean_object* v_00_u03c1_20_, lean_object* v_inst_21_, lean_object* v___x_22_, lean_object* v___x_23_, lean_object* v___y_24_, lean_object* v___y_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_tgl__kernel_TGLExt_gibbs___redArg___lam__2(v_a_19_, v_00_u03c1_20_, v_inst_21_, v___x_22_, v___x_23_, v___y_24_, v___y_25_);
lean_dec_ref(v___x_23_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_tgl__kernel_TGLExt_gibbs___redArg(lean_object* v_inst_27_, lean_object* v_00_u03c1_28_, lean_object* v_a_29_){
_start:
{
lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___f_32_; lean_object* v___x_33_; 
v___x_30_ = lp_mathlib_Complex_instAddCommMonoid;
v___x_31_ = lp_mathlib_Complex_instMul;
lean_inc(v_inst_27_);
v___f_32_ = lean_alloc_closure((void*)(lp_tgl__kernel_TGLExt_gibbs___redArg___lam__2___boxed), 7, 5);
lean_closure_set(v___f_32_, 0, v_a_29_);
lean_closure_set(v___f_32_, 1, v_00_u03c1_28_);
lean_closure_set(v___f_32_, 2, v_inst_27_);
lean_closure_set(v___f_32_, 3, v___x_31_);
lean_closure_set(v___f_32_, 4, v___x_30_);
v___x_33_ = lp_mathlib_Matrix_trace___redArg(v_inst_27_, v___x_30_, v___f_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_tgl__kernel_TGLExt_gibbs(lean_object* v_n_34_, lean_object* v_inst_35_, lean_object* v_00_u03c1_36_, lean_object* v_a_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_tgl__kernel_TGLExt_gibbs___redArg(v_inst_35_, v_00_u03c1_36_, v_a_37_);
return v___x_38_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_tgl__kernel_TGLExt_LeftRight(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_tgl__kernel_TGLExt_FiniteTomita(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_tgl__kernel_TGLExt_LeftRight(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
