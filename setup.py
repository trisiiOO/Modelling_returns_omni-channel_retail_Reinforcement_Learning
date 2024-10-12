from setuptools import setup, find_packages

setup(
    name='omni_channel_retail_simulation',
    version='0.1.0',
    author='Tristan Kruse',
    description='A PPO-based simulation for Omni-Channel Retailer decision-making with reinforcement learning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/trisiiOO/Modelling_returns_omni-channel_retail_Reinforcement_Learning',
    packages=find_packages(),
    py_modules=['main'],
    install_requires=[
        'pandas>=1.0.0',
        'numpy>=1.19.0',
        'tensorflow>=2.0.0',
        'gym>=0.17.0',
        'scipy>=1.4.0',
        'keras>=2.0.0',
        'pytest>=6.0.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'retail_sim=main:main',
        ],
    },
)
