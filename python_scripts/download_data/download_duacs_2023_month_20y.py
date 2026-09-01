import copernicusmarine

dataset_id = "c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1M-m"

output_directory = "/Odyssey/public/glorys/reanalysis"

get_data = copernicusmarine.subset(
    dataset_id=dataset_id,
    output_directory = output_directory,
    variables=["sla"],
    minimum_longitude=-180,
    maximum_longitude=180,
    minimum_latitude=-80,
    maximum_latitude=90,
    start_datetime="2000-01-01T00:00:00",
    end_datetime="2019-12-31T00:00:00",
)