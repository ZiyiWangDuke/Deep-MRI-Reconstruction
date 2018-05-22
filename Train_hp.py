import click

from dl_research.projects.degrade.scripts.slices import (
    MCD1_METADATA_CSV,
    MCD2_METADATA_CSV,
)

from dl_research.papers.automap.data import axial_non_dw_filter_256, load_data
import os, pdb


@click.command()
def fit():
    # load data 
    data_train, data_valid = load_data(
        data_csvs=[(MCD1_METADATA_CSV, os.path.dirname(MCD1_METADATA_CSV))],
        df_filter=axial_non_dw_filter_256,
    )

    pdb.set_trace()

    
if __name__ == '__main__':
    fit()



