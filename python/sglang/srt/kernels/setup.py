from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='quest_attention',
    ext_modules=[
        CUDAExtension('quest_attention', [
            'quest_attention.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })