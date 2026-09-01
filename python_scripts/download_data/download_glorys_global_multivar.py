import copernicusmarine

dataset_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
output_directory = "/Odyssey/public/glorys/reanalysis/multivar/"

get_data = copernicusmarine.subset(
    dataset_id=dataset_id,
    output_directory = output_directory,
    username="phaslee1",
    password="AfH5$3aXA!EnRa8DtS4f",
    variables=["thetao", "zos", "mlotst"],
    minimum_longitude=-180,
    maximum_longitude=180,
    minimum_latitude=-80,
    maximum_latitude=90,
    start_datetime="2010-01-01T00:00:00",
    end_datetime="2019-12-31T00:00:00",
    minimum_depth=0.49402499198913574,
    maximum_depth=0.49402499198913574,
)