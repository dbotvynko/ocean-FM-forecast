import copernicusmarine

dataset_id = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
output_directory = "/DATASET/GLORYS12/mercator_forecast/"

get_data = copernicusmarine.subset(
    dataset_id=dataset_id,
    output_directory = output_directory,
    username="phaslee1",
    password="AfH5$3aXA!EnRa8DtS4f",
    variables=["zos"],
    minimum_longitude=-180,
    maximum_longitude=180,
    minimum_latitude=-80,
    maximum_latitude=90,
    start_datetime="2023-09-01T00:00:00",
    end_datetime="2023-09-30T00:00:00",
)