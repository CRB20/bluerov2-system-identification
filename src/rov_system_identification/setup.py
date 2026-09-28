from setuptools import find_packages, setup

package_name = "rov_system_identification"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/identification.yaml"]),
        ("share/" + package_name + "/launch", ["launch/collector.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="ROV 6-DOF physical data collection and system identification",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "collect_physical_data = rov_system_identification.collect_physical_data:main",
            "run_identification = rov_system_identification.run_identification:main",
            "compare_models = rov_system_identification.compare_models:main",
            "plot_results = rov_system_identification.plot_results:main",
            "run_full_pipeline = rov_system_identification.run_full_pipeline:main",
            "frequency_analysis = rov_system_identification.frequency_analysis:main",
            "per_dof_selection = rov_system_identification.per_dof_selection:main",
            "analyze_disturbances = rov_system_identification.disturbance_analysis:main",
        ],
    },
)
