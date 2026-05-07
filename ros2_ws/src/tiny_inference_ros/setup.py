from glob import glob
from setuptools import setup


package_name = "tiny_inference_ros"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.json")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="tiny-inference",
    maintainer_email="todo@example.com",
    description="Bridge tiny-inference JSON plans into a scripted Panda pick/place demo.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "scripted_pick_place = tiny_inference_ros.scripted_pick_place_node:main",
        ],
    },
)
