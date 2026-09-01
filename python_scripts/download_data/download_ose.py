import copernicusmarine

dataset_id = "cmems_obs-sl_glo_phy-ssh_nrt_s6a-hr-l3-duacs_PT1S"
output_directory = "/DATASET/OCB_traces/get/"


get_data = copernicusmarine.subset(
    dataset_id=dataset_id,
    output_directory = output_directory,
    username="dzhu1",
    password="kJ5U19v9v793p1PP23z97rHWhK3yL7cGgf86pS5u5ckFMkbAS217EDwB5fG8MnUQ",
    variables=["dac", "internal_tide", "lwe", "mdt", "ocean_tide", "sla_filtered", "sla_unfiltered"],
    minimum_longitude=-66,
    maximum_longitude=-54,
    minimum_latitude=32,
    maximum_latitude=44,
    start_datetime="2022-01-01T00:00:00",
    end_datetime="2022-12-31T00:00:00",
)

"""
get_data = copernicusmarine.get(
    dataset_id=dataset_id,
    output_directory = output_directory,
    username="dzhu1",
    password="kJ5U19v9v793p1PP23z97rHWhK3yL7cGgf86pS5u5ckFMkbAS217EDwB5fG8MnUQ",
    
)
"""