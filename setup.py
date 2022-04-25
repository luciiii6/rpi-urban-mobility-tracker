from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open('requirements.txt') as f:
    required = f.read().splitlines()
    
setup(
    name='umt',
    version='0.0.3',
    author='Nathan A. Rooy, Suvaila Lucian-Ioan',
    author_email='nathanrooy@gmail.com, luci.suvaila@gmail.com',
    url='https://github.com/luciiii6/rpi-urban-mobility-tracker',
    description='A way to count people using machine learning',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['umt'],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    install_requires=required,
    entry_points={
        'console_scripts': [
            'umt = umt.app:main'
        ]
    },
    package_data={
    	'umt':[
    		'models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/labelmap.txt',
    		'models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/detect.tflite',
    		'models/tpu/mobilenet_ssd_v2_coco_quant/coco_labels.txt',
    		'models/tpu/mobilenet_ssd_v2_coco_quant/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite',
            'deep_sort/*'
    	]
    },
)
