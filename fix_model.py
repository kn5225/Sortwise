import h5py
import json

def remove_groups(config):
    """Recursively remove 'groups' key from all layer configs."""
    if isinstance(config, dict):
        config.pop('groups', None)
        for v in config.values():
            remove_groups(v)
    elif isinstance(config, list):
        for item in config:
            remove_groups(item)

with h5py.File('keras_model_original.h5', 'r') as f_in:
    model_config = json.loads(f_in.attrs['model_config'])  # Read and patch the model config
    remove_groups(model_config)
    with h5py.File('keras_model_fixed.h5', 'w') as f_out: # Copy everything to a new file with the patched config
        f_out.attrs['model_config'] = json.dumps(model_config) # Write patched config
        for key, val in f_in.attrs.items():        # Copy all other attributes
            if key != 'model_config':
                f_out.attrs[key] = val
        def copy_group(src, dst):  # Copy all weight data
            for key in src.keys():
                if isinstance(src[key], h5py.Group):
                    grp = dst.require_group(key)
                    copy_group(src[key], grp)
                else:
                    dst.create_dataset(key, data=src[key][()])
                for attr_key, attr_val in src[key].attrs.items():
                    dst[key].attrs[attr_key] = attr_val
        copy_group(f_in, f_out)

print("Done! keras_model_fixed.h5 is ready.")
