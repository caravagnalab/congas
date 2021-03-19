def export_switch(x):
    export = ["CNV_probabilities"]
    if 'data_rna' in x._data:
        export.append("mixture_weights_rna")
        export.append("segment_factor_rna")
        if x._params['likelihood_rna'] == "NB":
            export.append("NB_size_rna")
        if x._params['likelihood_rna'] in ["Normal", "Gaussian"]:
            export.append("norm_sd_rna")
    if 'data_atac' in x._data:
        export.append("mixture_weights_atac")
        export.append("segment_factor_atac")
        if x._params['likelihood_atac'] == "NB":
            export.append("NB_size_atac")
        if x._params['likelihood_atac'] in ["Normal", "Gaussian"]:
            export.append("norm_sd_atac")

    return export
