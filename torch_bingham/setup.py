"""
   torch bingham module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Haowen Deng
"""

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

extra_objects = []
module = CppExtension(
    'torch_bingham', [
    'external/bingham/bingham.cpp',
    'external/bingham/bingham_constants.cpp',
    'external/bingham/tetramesh.cpp',
    'external/bingham/octetramesh.cpp',
    'external/bingham/hypersphere.cpp',
    'external/bingham/util.cpp',
    'ops.cpp'],
    include_dirs=['external/bingham/include', '.'],
    libraries=['m'],
    extra_objects = extra_objects,
)

setup(
    name='torch_bingham',
    ext_modules=[module],
    cmdclass={'build_ext': BuildExtension})
