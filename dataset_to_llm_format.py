import pandas as pd

import argparse
import pathlib
import warnings

def build():
    default_help = "(default: %(default)s)"
    prs = argparse.ArgumentParser()
    prs.add_argument('--in', dest='_in', type=pathlib.Path, required=True, nargs='+', action='append',
                     default=None, help=f"CSVs to ingest (and combine, if multiple are present!) {default_help}")
    prs.add_argument('--out', dest='_out', type=pathlib.Path, required=True,
                     help=f"Output CSV path {default_help}")
    prs.add_argument('--remap-column', type=str, nargs='*', action='append', default=None,
                     help=f"Rename output columns 'FROM:TO' where FROM column in --in CSVs are renamed to TO column in --out CSV {default_help}")
    prs.add_argument('--drop-columns', type=str, nargs='*', action='append', default=None,
                     help=f"Drop columns from --in CSV prior to export {default_help}")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs=build()
    if args is None:
        args = prs.parse_args()
    # Validate files
    _in = []
    mia = []
    args._in = args._in[0]
    for f in args._in:
        if not f.exists():
            mia.append(f)
        else:
            _in.append(f)
    if len(mia) > 0:
        raise IOError(f"Did not find input files: {mia}")
    args._in = _in
    if args._out.exists():
        warnings.warn(f"Output file {args._out} will be overwritten", UserWarning)
    if args.drop_columns is None:
        args.drop_columns = []
    else:
        args.drop_columns = args.drop_columns[0]
    # Handle mapping
    maps = []
    bad = []
    if args.remap_column is None:
        args.remap_column = []
    else:
        args.remap_column = args.remap_column[0]
    for m in args.remap_column:
        if m.count(':') != 1:
            bad.append(m)
        else:
            maps.append(m.split(':',1))
    if len(bad) > 0:
        warnings.warn(f"Dropping bad mappings (format: 'FROM:TO'): {bad}")
    args.remap_column = maps

    return args

def main():
    args = parse()
    # Load and combine all requested in CSVs
    csvs = []
    for c in args._in:
        csv = pd.read_csv(c)
        if 'size' not in csv.columns:
            size = None
            if '_xl_' in c.stem.lower():
                size = 'XL'
            elif '_ml_' in c.stem.lower():
                size = 'ML'
            elif '_sm_' in c.stem.lower():
                size = 'SM'
            elif '_l_' in c.stem.lower():
                size = 'L'
            elif '_m_' in c.stem.lower():
                size = 'M'
            elif '_s_' in c.stem.lower():
                size = 'S'
            if size is None:
                raise ValueError(f'Failed to determine size to fill in for {c}!')
            csv.insert(0,'size', [size]*len(csv))
        csvs.append(csv)
    # Infer size if not present
    csvs = pd.concat(csvs)
    # Drop and rename columns
    csvs.drop(columns=args.drop_columns, inplace=True)
    csvs.rename(columns=dict((k,v) for (k,v) in args.remap_column),
                inplace=True)
    # Specific to my use case for now
    tf_cols = [_ for _ in csvs.columns if len(set(csvs[_])) == 2 and ' ' in set(csvs[_])]
    for col in tf_cols:
        tfdata = [True if _ != ' ' else False for _ in csvs[col]]
        csvs.loc[:,col] = tfdata
    # Output
    csvs.to_csv(args._out, index=False)

if __name__ == '__main__':
    main()

