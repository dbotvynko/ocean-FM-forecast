import copernicusmarine

dataset_id = "c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D"
output_directory = "/DATASET/GLORYS12/reanalysis"

get_data = copernicusmarine.subset(
    dataset_id=dataset_id,
    output_directory = output_directory,
    variables=["adt", "sla", "err_sla"],
    minimum_longitude=-180,
    maximum_longitude=180,
    minimum_latitude=-80,
    maximum_latitude=90,
    start_datetime="2023-01-01T00:00:00",
    end_datetime="2023-06-07T00:00:00",
)