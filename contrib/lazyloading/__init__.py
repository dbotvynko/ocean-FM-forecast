import torch

class LazyXrDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ds,
        patch_dims,
        domain_limits=None,
        strides=None,
        postpro_fn=None,
        noise_type=None,
        noise=None,
        noise_spatial_perturb=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.return_coords = False
        self.postpro_fn = postpro_fn
        self.ds = ds.sel(**(domain_limits or {}))
        self.patch_dims = patch_dims
        self.strides = strides or {}
        _dims = ("variable",) + tuple(k for k in self.ds.dims)
        _shape = (2,) + tuple(self.ds[k].shape[0] for k in self.ds.dims)
        ds_dims = dict(zip(_dims, _shape))
        # ds_dims = dict(zip(self.ds.dims, self.ds.shape))
        

        self.ds_size = {
            dim: max(
                (ds_dims[dim] - patch_dims[dim]) // strides.get(dim, 1) + 1,
                0,
            )
            for dim in patch_dims
        }
        self._rng = np.random.default_rng()
        self.noise = noise
        self.noise_spatial_perturb = noise_spatial_perturb

        if noise_type is not None:
            self.noise_type = noise_type
        else:
            self.noise_type = 'uniform-constant'

        self.mask = kwargs.get("mask")

        print(self.noise_type,flush=True)

    def __len__(self):
        size = 1
        for v in self.ds_size.values():
            size *= v
        return size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_coords(self):
        self.return_coords = True
        coords = []
        try:
            for i in range(len(self)):
                coords.append(self[i])
        finally:
            self.return_coords = False
            return coords

    def __getitem__(self, item):
        sl = {}
        _zip = zip(
            self.ds_size.keys(), np.unravel_index(item, tuple(self.ds_size.values()))
        )

        for dim, idx in _zip:
            sl[dim] = slice(
                self.strides.get(dim, 1) * idx,
                self.strides.get(dim, 1) * idx + self.patch_dims[dim],
            )

        if self.mask is not None:
            start, stop = sl["time"].start % 365, sl["time"].stop % 365
            if start > stop:
                start -= stop
                stop = None
            sl_mask = sl.copy()
            sl_mask["time"] = slice(start, stop)

            da = self.ds.isel(**sl)

            item = (
                da.to_dataset(name="tgt")
                .assign(input=da.where(self.mask.isel(**sl_mask).values))
                .to_array()
                .sortby("variable")
            )
        else:
            item = self.ds.isel(**sl)

        if self.return_coords:
            return item.coords.to_dataset()[list(self.patch_dims)]

        item = item.data.astype(np.float32)

        #if self.noise:
        #    noise = np.tile(
        #        self._rng.uniform(-self.noise, self.noise, item[0].shape), (2, 1, 1, 1)
        #    ).astype(np.float32)
        #    item = item + noise

        if self.noise is not None:

            if self.noise_type ==  'uniform-constant' :
                #noise =  self._rng.uniform(-self.noise, self.noise, item[0].shape).astype(np.float32)
                #item[0] = item[0] + noise
                noise =  self._rng.uniform(-self.noise, self.noise, item.shape).astype(np.float32)

                item = item + noise
            elif self.noise_type ==  'gaussian+uniform' :
                scale = self._rng.uniform(0. , self.noise, 1).astype(np.float32)
                noise = self._rng.normal(0., 1. , item[0].shape).astype(np.float32)

                item[0] = item[0] + scale * noise
            elif self.noise_type ==  'spatial-perturb' :
                # patch dimensions
                N = item[0].shape[1]
                M = item[0].shape[2]
                T = item[0].shape[0]

                # parameters of the Gaussian Process
                L = 5.0  # Spatial correlation length
                T_corr = 20.0  # Temporal correlation length
                sigma  = self._rng.uniform(0. , self.noise_spatial_perturb, 1).astype(np.float32)

                # generate space-time random perturbations
                w  = np.matmul( np.hanning( N ).reshape(N,1) , np.hanning( M ).reshape(1,M) )
                dx = w * generate_correlated_fields_np(N, M, L, T_corr, sigma,T)
                dy = w * generate_correlated_fields_np(N, M, L, T_corr, sigma,T)

                # compute warped fields from reference field and associated error map
                warped_field = np.zeros_like(field)

                # Perform warping
                for ii in range(T):
                    warped_field[ii,:,:] = warp_field_np(item[1][ii,:,:], dx[ii,:,:], dy[ii,:,:])
                
                # residual error
                error = item[1][ii,:,:] - warped_field

                # apply mask
                noise = np.where( np.isnan(item[0]) , np.nan, error )
                print("..... std of the simulated noise : %.3f"%np.nanstd(noise) )
                
                # adding a white noise
                scale  = self._rng.uniform(0. , self.noise, 1).astype(np.float32)
                wnoise = scale * self._rng.normal(0., 1. , item[0].shape).astype(np.float32)

                item[0] = item[0] + noise + wnoise
        
        if self.postpro_fn is not None:
            return self.postpro_fn(item)
        return item

def load_glorys12_data_on_fly_inp(
    tgt_path,
    inp_path,
    tgt_var="zos",
    inp_var="input",
):
 
    print('..... Start lazy loading',flush=True)

    isel = None  # dict(time=slice(-365 * 2, None))

    tgt = (
        xr.open_dataset(tgt_path)[tgt_var]
        .isel(isel)
        #.rename(latitude="lat", longitude="lon")
    )

    print('..... GT dataset')

    if list(tgt.coords)[1] == 'latitude' :
        tgt = tgt.rename(latitude="lat", longitude="lon")

    print(tgt,flush=True)

    inp = (
        xr.open_dataset(inp_path)[inp_var]
        .isel(isel)
        #.rename(latitude="lat", longitude="lon")
    )

    if list(inp.coords)[1] == 'latitude' :
        inp = inp.rename(latitude="lat", longitude="lon")

    print('..... Obs dataset')
    print('..... NB: only the mask of the obs data might be used by the datamodule')
    print(inp,flush=True)

    return tgt, inp