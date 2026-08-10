// Lean compiler output
// Module: TGLExt.CondExpect
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
extern lean_object* lp_mathlib_Complex_instZero;
lean_object* lp_mathlib_Matrix_diag(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_diagonal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_tgl__kernel_TGLExt_diagExpect___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_tgl__kernel_TGLExt_diagExpect(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_tgl__kernel_TGLExt_eD___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_tgl__kernel_TGLExt_eD(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_tgl__kernel_TGLExt_diagExpect___redArg(lean_object* v_inst_1_, lean_object* v_x_2_, lean_object* v_a_3_, lean_object* v_a_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; 
v___x_5_ = lp_mathlib_Complex_instZero;
v___x_6_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_diag), 4, 3);
lean_closure_set(v___x_6_, 0, lean_box(0));
lean_closure_set(v___x_6_, 1, lean_box(0));
lean_closure_set(v___x_6_, 2, v_x_2_);
v___x_7_ = lp_mathlib_Matrix_diagonal___redArg(v_inst_1_, v___x_5_, v___x_6_, v_a_3_, v_a_4_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_tgl__kernel_TGLExt_diagExpect(lean_object* v_n_8_, lean_object* v_inst_9_, lean_object* v_x_10_, lean_object* v_a_11_, lean_object* v_a_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_tgl__kernel_TGLExt_diagExpect___redArg(v_inst_9_, v_x_10_, v_a_11_, v_a_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_tgl__kernel_TGLExt_eD___redArg(lean_object* v_inst_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lean_alloc_closure((void*)(lp_tgl__kernel_TGLExt_diagExpect), 5, 2);
lean_closure_set(v___x_15_, 0, lean_box(0));
lean_closure_set(v___x_15_, 1, v_inst_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_tgl__kernel_TGLExt_eD(lean_object* v_n_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lean_alloc_closure((void*)(lp_tgl__kernel_TGLExt_diagExpect), 5, 2);
lean_closure_set(v___x_18_, 0, lean_box(0));
lean_closure_set(v___x_18_, 1, v_inst_17_);
return v___x_18_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_tgl__kernel_TGLExt_LeftRight(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_tgl__kernel_TGLExt_CondExpect(uint8_t builtin) {
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
