library(TED)
require(data.table)
require(tidyr)

read_and_format <- function(in_file){
    
    sc_matr = fread(in_file)
    cell_ids = colnames(sc_matr)[2:ncol(sc_matr)]
    sc_matr = data.frame(sc_matr)
    
    # format the column names
    colnames(sc_matr)[1] = "full_gene_ids"
    
    # take all value columns
    count_matr = sc_matr[,2:ncol(sc_matr)]
    
    # format the data for BayesPrism
    ens_gene_ids = sc_matr$full_gene_ids
    
    count_matr_t = t(count_matr)
    colnames(count_matr_t) = ens_gene_ids
    rownames(count_matr_t) = cell_ids
    
    return(count_matr_t)
    
}


read_and_format_train <- function(in_dir, file_id_test){

    # get all the sig files generated
    all_files = list.files(in_dir, full.names=T)
    all_files = grep("_cybersort_sig.tsv.gz", all_files, value=T)
    all_files = grep(file_id_test, all_files, invert=T, value=T)
    all_files = grep("MGH85", all_files, invert=T, value=T)

    count_matr_t = NA
    for(curr_file in all_files){
        curr_matr = read_and_format(curr_file)
        if(is.na(count_matr_t)){
            count_matr_t = curr_matr
        }else{
            col_ids = intersect(colnames(curr_matr), colnames(count_matr_t))
            count_matr_t = count_matr_t[,col_ids]
            curr_matr = curr_matr[,col_ids]
            count_matr_t = rbind(count_matr_t, curr_matr)
        }
    }
    
    return(count_matr_t)
    
}


run_bayes_prism <- function(in_dir, out_dir, file_id_test, num_samp, ncores){
    sc_ref = read_and_format_train(in_dir, file_id_test)
    
    bulk_matr_file = paste0(in_dir, file_id_test, "_cybersort_mix.tsv.gz")
    bulk_ref = read_and_format(bulk_matr_file)
    bulk_ref = data.frame(bulk_ref)
    rownames(bulk_ref) = paste0("samp_", bulk_ref$gene_ids)
    bulk_ref <- subset(bulk_ref, select = -c(gene_ids))
    bulk_ref = as.matrix(bulk_ref)
    
    sc_ref_filtered = cleanup.genes(sc_ref,
                                    species="hs",
                                    gene.type=c("RB","chrM","chrX","chrY"),
                                    input.type = "scRNA",
                                    exp.cells = 5)
    dim(sc_ref_filtered)
    dim(bulk_ref)
    # only run up to the max number of samples requested
    num_samp = min(nrow(bulk_ref), num_samp)

    bp_out = run.Ted(ref.dat=sc_ref_filtered,
                     X = round(bulk_ref[1:num_samp,]),
                     cell.type.labels = rownames(sc_ref_filtered),
                     tum.key="Malignant",
                     input.type = "scRNA",
                     n.cores=ncores)
    
    curr_out_file = paste0(out_dir, "/test-", file_id_test, "-bp_", num_samp, "_prop.tsv")
    cell_frac = bp_out$res$final.gibbs.theta
    write.table(cell_frac, curr_out_file, sep="\t", quote=F, row.names = F)
    
    out_file = paste0(out_dir, "/test-", file_id_test, "-bp_", num_samp, ".rds")
    saveRDS(bp_out, out_file)
    
}

#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
in_dir = args[1]
out_dir = args[2]
file_id_test = args[3]
num_samp = as.numeric(args[4])
ncores = as.numeric(args[5])

print("in_dir:")
print(in_dir)

print("out_dir:")
print(out_dir)

print("file_id_test:")
print(file_id_test)

print("num_samp:")
print(num_samp)

print("ncores:")
print(ncores)

run_bayes_prism(in_dir, out_dir, file_id_test, num_samp, ncores)
