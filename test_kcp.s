
.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
.text
.globl test_nowgp
.p2align 8
.type test_nowgp,@function
test_nowgp:
  s_endpgm
.Lfunc_end:
  .size test_nowgp, .Lfunc_end-test_nowgp
.rodata
.p2align 6
.amdhsa_kernel test_nowgp
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 0
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_workgroup_processor_mode 1
.end_amdhsa_kernel
