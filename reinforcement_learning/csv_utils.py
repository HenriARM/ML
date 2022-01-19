import copy
import os
import time
import logging
import sys
import traceback
import numpy as np
from file_utils import FileUtils


class CsvUtils:

    @staticmethod
    def add_hparams(sequence_dir, sequence_name, run_name, args_dict, metrics_dict, global_step):
        # create sequence file and for each run separate
        if not os.path.exists(sequence_dir):
            os.makedirs(sequence_dir)
        sequence_path = os.path.join(sequence_dir, f'{sequence_name}.csv')
        run_path = os.path.join(sequence_dir, f'{run_name}.csv')
        if not os.path.exists(sequence_path):
            open(file=sequence_path, mode='a').close()
        if not os.path.exists(run_path):
            open(file=run_path, mode='a').close()

        # edit args and metrics dict
        args_dict = copy.copy(args_dict)
        metrics_dict = copy.copy(metrics_dict)
        for each_dict in [args_dict, metrics_dict]:
            for key in list(each_dict.keys()):
                if not isinstance(each_dict[key], float) and \
                        not isinstance(each_dict[key], int) and \
                        not isinstance(each_dict[key], str) and \
                        not isinstance(each_dict[key], np.float) and \
                        not isinstance(each_dict[key], np.int) and \
                        not isinstance(each_dict[key], np.float32):
                    del each_dict[key]

        # add all params into sequence and run files
        for path_csv in [sequence_path, run_path]:
            with open(path_csv, 'r+') as outfile:
                FileUtils.lock_file(outfile)
                lines_all = outfile.readlines()
                lines_all = [it.replace('\n', '').split(',') for it in lines_all if ',' in it]
                if len(lines_all) == 0:  # or len(lines_all[0]) < 2:
                    headers = ['step'] + list(args_dict.keys()) + list(metrics_dict.keys())
                    headers = [str(it).replace(',', '_') for it in headers]
                    lines_all.append(headers)

                values = [global_step] + list(args_dict.values()) + list(metrics_dict.values())
                values = [str(it).replace(',', '_') for it in values]
                if path_csv == run_path:
                    lines_all.append(values)
                else:
                    # sequence file
                    existing_line_idx = -1
                    args_values = list(args_dict.values())
                    args_values = [str(it).replace(',', '_') for it in args_values]
                    for idx_line, line in enumerate(lines_all):
                        if len(line) > 1:
                            is_match = True
                            for idx_arg in range(len(args_values)):
                                if line[idx_arg + 1] != args_values[idx_arg]:
                                    is_match = False
                                    break
                            if is_match:
                                existing_line_idx = idx_line
                                break
                    if existing_line_idx >= 0:
                        lines_all[existing_line_idx] = values
                    else:
                        lines_all.append(values)

                outfile.truncate(0)
                outfile.seek(0)
                outfile.flush()
                rows = [','.join(it) for it in lines_all]
                rows = [it for it in rows if len(it.replace('\n', '').strip()) > 0]
                outfile.write('\n'.join(rows).strip())
                outfile.flush()
                os.fsync(outfile)
                FileUtils.unlock_file(outfile)
