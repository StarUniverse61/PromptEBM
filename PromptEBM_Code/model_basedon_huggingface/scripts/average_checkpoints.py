#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import collections
import torch
import os
import re

def checkpoint_paths(path, pattern=r'checkpoint-(\d+)-epoch-(\d+)'):
    """Retrieves all checkpoints found in `path` directory.

    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order.
    """
    pt_regexp = re.compile(pattern)
    files = os.listdir(path)

    entries = []
    for i, f in enumerate(files):
        m = pt_regexp.fullmatch(f)
        if m is not None:
            idx = int(m.group(2)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
            #print(m.group(0))
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]

def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs: An iterable of string paths of checkpoints to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    for f in inputs:
        state = torch.load(
            f+'/pytorch_model.bin',
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        #model_params = state['model']
        model_params = state
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state['model'] = averaged_params
    return new_state


def last_n_checkpoints(paths, n, update_based, upper_bound=None):
    assert len(paths) == 1
    path = paths[0]
    if update_based:
        pt_regexp = re.compile(r'checkpoint-(\d+)-epoch-(\d+)')
    else:
        pt_regexp = re.compile(r'checkpoint-(\d+)-epoch-(\d+)')
    files = os.listdir(path)

    entries = []
    for f in files:
        m = pt_regexp.fullmatch(f)
        if m is not None:
            sort_key = int(m.group(1))
            if upper_bound is None or sort_key <= upper_bound:
                entries.append((sort_key, m.group(0)))
    if len(entries) < n:
        raise Exception('Found {} checkpoint files but need at least {}', len(entries), n)
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)[:n]]

def main():
    parser = argparse.ArgumentParser(
        description='Tool to average the params of input checkpoints to '
                    'produce a new checkpoint',
    )
    # fmt: off
    parser.add_argument('--inputs', required=True, nargs='+',
                        help='Input checkpoint file paths.')
    parser.add_argument('--output', required=True, metavar='FILE',
                        help='Write the new checkpoint containing the averaged weights to this path.')
    num_group = parser.add_mutually_exclusive_group()
    num_group.add_argument('--num-epoch-checkpoints', type=int,
                           help='if set, will try to find checkpoints with names checkpoint_xx.pt in the path specified by input, '
                           'and average last this many of them.')
    num_group.add_argument('--num-update-checkpoints', type=int,
                           help='if set, will try to find checkpoints with names checkpoint_ee_xx.pt in the path specified by input, '
                           'and average last this many of them.')
    parser.add_argument('--checkpoint-upper-bound', type=int,
                        help='when using --num-epoch-checkpoints, this will set an upper bound on which checkpoint to use, '
                        'e.g., with --num-epoch-checkpoints=10 --checkpoint-upper-bound=50, checkpoints 41-50 would be averaged.')
    # fmt: on
    args = parser.parse_args()
    print(args)
    import shutil
    #file_names = os.listdir(args.inputs[0])
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    for filename in os.listdir(args.inputs[0]):
        if 'json' in filename or 'txt' in filename or 'training_args.bin' in filename:
            shutil.copyfile(os.path.join(args.inputs[0], filename), os.path.join(args.output, filename))
    num = None
    is_update_based = False
    if args.num_update_checkpoints is not None:
        num = args.num_update_checkpoints
        is_update_based = True
    elif args.num_epoch_checkpoints is not None:
        num = args.num_epoch_checkpoints

    assert args.checkpoint_upper_bound is None or args.num_epoch_checkpoints is not None, \
        '--checkpoint-upper-bound requires --num-epoch-checkpoints'
    assert args.num_epoch_checkpoints is None or args.num_update_checkpoints is None, \
        'Cannot combine --num-epoch-checkpoints and --num-update-checkpoints'
    all_checkpoint_files = checkpoint_paths(args.inputs[0])
    if num is not None:
        args.inputs = last_n_checkpoints(
            args.inputs, num, is_update_based, upper_bound=args.checkpoint_upper_bound,
        )
        print('averaging checkpoints: ', args.inputs)


    new_state = average_checkpoints(args.inputs)
    torch.save(new_state, args.output+'/pytorch_model.bin')
    '''
    remove all checkpoint files for saving spaces
    '''
    import shutil
    for checkpoint_file in all_checkpoint_files:
        print(f'checkpoing_file={checkpoint_file}')
        try:
            shutil.rmtree(checkpoint_file)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
        #os.rmdir(checkpoint_file) rmdir only remove empty folder
    print('Finished writing averaged checkpoint to {}.'.format(args.output))
if __name__ == '__main__':
    main()
