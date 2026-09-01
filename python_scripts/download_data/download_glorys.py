import copernicusmarine

dataset_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
output_directory = "/DATASET/GLORYS12/reanalysis"

get_data = copernicusmarine.subset(
    dataset_id=dataset_id,
    output_directory = output_directory,
    username="dzhu1",
    password="kJ5U19v9v793p1PP23z97rHWhK3yL7cGgf86pS5u5ckFMkbAS217EDwB5fG8MnUQ",
    dataset_version="202311",
    variables=["so", "thetao", "uo", "vo", "zos", "mlotst"],
    minimum_longitude=-66,
    maximum_longitude=-54,
    minimum_latitude=32,
    maximum_latitude=44,
    start_datetime="2010-01-01T00:00:00",
    end_datetime="2019-12-31T00:00:00",
    minimum_depth=0.49402499198913574,
    maximum_depth=0.49402499198913574,
)