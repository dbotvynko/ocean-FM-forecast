import copernicusmarine

dataset_id = "cmems_obs-sst_glo_phy_nrt_l4_P1D-m"
output_directory = "/Odyssey/public/nrt_sst/"

get_data = copernicusmarine.subset(
    dataset_id=dataset_id,
    output_directory = output_directory,
    username="phaslee1",
    password="AfH5$3aXA!EnRa8DtS4f",
    variables=["analysed_sst"],
    minimum_longitude=-180,
    maximum_longitude=180,
    minimum_latitude=-80,
    maximum_latitude=90,
    start_datetime="2023-01-01T00:00:00",
    end_datetime="2023-12-31T00:00:00",
    credentials_file='/users/local/p24hasle/.copernicusmarine'
)