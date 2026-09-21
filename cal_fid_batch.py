from calculate_fid import calculate_fid_given_paths


fid_reference_file = '/video_ssd/kongzishang/xutongda/git/REPA-E/exps/VIRTUAL_imagenet256_labeled.npz'

fid_num = 50000

sample_folder_dirs = [
    'samples/sit-xl-flux2vae-0424-400k_0400000_cfg3.5-0.0-1.0',
    'samples/sit-xl-flux2vae-0424-400k_0400000_cfg3.75-0.0-1.0',
    'samples/sit-xl-flux2vae-0424-400k_0400000_cfg4.0-0.0-1.0',
    'samples/sit-xl-flux2vae-0424-400k_0400000_cfg4.25-0.0-1.0',
    'samples/sit-xl-flux2vae-0424-400k_0400000_cfg4.5-0.0-1.0',
    'samples/sit-xl-flux2vae-0424-400k_0400000_cfg4.75-0.0-1.0',
    'samples/sit-xl-flux2vae-0424-400k_0400000_cfg5.0-0.0-1.0',
    'samples/sit-xl-flux2vae-0424-400k_0400000_cfg5.25-0.0-1.0'
    'samples/sit-xl-flux2vae-0424-400k_0400000_cfg5.75-0.0-1.0',
    'samples/sit-xl-flux2vae-0424-400k_0400000_cfg6.0-0.0-1.0',
]


def main():
    results = []

    for sample_folder_dir in sample_folder_dirs:
        print('=' * 80)
        print(f'Calculating FID with {fid_num} samples')
        print(f'Sample folder: {sample_folder_dir}')

        try:
            fid = calculate_fid_given_paths(
                [fid_reference_file, sample_folder_dir],
                batch_size=50,
                dims=2048,
                device='cuda',
                num_workers=4,
                sp_len=fid_num
            )

            print(f'fid = {fid}')
            results.append((sample_folder_dir, fid))

        except Exception as e:
            print(f'Error when calculating FID for: {sample_folder_dir}')
            print(f'Error message: {e}')
            results.append((sample_folder_dir, None))

    print('\n' + '=' * 80)
    print('Final FID Results')
    print('=' * 80)

    for folder, fid in results:
        print(f'{folder}\t{fid}')


if __name__ == '__main__':
    main()