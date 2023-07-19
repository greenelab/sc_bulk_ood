library(TED)
require(data.table)
require(tidyr)

read_and_format <- function(in_file){
    
    sc_matr = fread(in_file, header=T)
    cell_ids = colnames(sc_matr)[2:ncol(sc_matr)]
    sc_matr = data.frame(sc_matr)
    
    # format the column names
    colnames(sc_matr)[1] = "ensembl"
    #colnames(sc_matr)[1] = "full_gene_ids"
    #sc_matr = separate(data = sc_matr, col = full_gene_ids, into = c("ensembl", "hgnc"), sep="_")
    
    # take all value columns
    count_matr = sc_matr[,2:ncol(sc_matr)]
    #count_matr = sc_matr[,3:ncol(sc_matr)]
    
    # format the data for BayesPrism
    ens_gene_ids = sc_matr$ensembl
    
    count_matr_t = t(count_matr)
    colnames(count_matr_t) = ens_gene_ids
    rownames(count_matr_t) = cell_ids
    
    return(count_matr_t)
    
}

read_combine_files <- function(in_dir, file_id_train){

    labeled_ids = paste(file_id_train, 1:4, sep="_")

    unlabeled_ids = paste(file_id_train, 1:9, sep="_")
    unlabeled_ids = c(unlabeled_ids,  paste(file_id_train, "test_6", sep="_"))

    sc_ref_total = NA
    for(curr_lab_id in labeled_ids){
        sc_matr_file = paste0(in_dir, curr_lab_id, "_cybersort_sig.tsv.gz")
        sc_ref = read_and_format(sc_matr_file)

        if(all(is.na(sc_ref_total))){
            sc_ref_total = sc_ref
        }else{
            sc_ref_total = rbind(sc_ref_total, sc_ref)
        }
    }

    bulk_ref_total = NA
    for(curr_unlab_id in unlabeled_ids){


        bulk_matr_file = paste0(in_dir, curr_unlab_id, "_cybersort_mix.tsv.gz")
        bulk_ref = read_and_format(bulk_matr_file)
        bulk_ref = data.frame(bulk_ref)
        rownames(bulk_ref) = paste0("samp_", rownames(bulk_ref))
        bulk_ref = as.matrix(bulk_ref)

        if(all(is.na(bulk_ref_total))){
            bulk_ref_total = bulk_ref
        }else{
            bulk_ref_total = rbind(bulk_ref_total, bulk_ref)
        }
    }

    return(list(sc_ref_total, bulk_ref_total))


}

run_bayes_prism <- function(in_dir, out_dir, file_id_train, file_id_test, num_samp, ncores){

    res = read_combine_files(in_dir, file_id_train)
    sc_ref = res[[1]]
    bulk_ref = res[[2]]
    
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
                     input.type = "scRNA",
                     n.cores=ncores,
                     first.gibbs.only=TRUE)
    
    curr_out_file = paste0(out_dir, "/train-", file_id_train, "-test-", file_id_test, "-bp_", num_samp, "_prop.tsv")
    cell_frac = bp_out$res$final.gibbs.theta
    write.table(cell_frac, curr_out_file, sep="\t", quote=F, row.names = F)
    
    curr_out_file = paste0(out_dir, "/train-", file_id_train, "-test-", file_id_test, "-bp_", num_samp, "_init.tsv")
    cell_frac = bp_out$res$first.gibbs.res$gibbs.theta
    write.table(cell_frac, curr_out_file, sep="\t", quote=F, row.names = F)

    out_file = paste0(out_dir, "/train-", file_id_train, "-test-", file_id_test, "-bp_", num_samp, ".rds")
    saveRDS(bp_out, out_file)
    
}

#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
in_dir = args[1]
out_dir = args[2]
file_id_train = args[3]
file_id_test = args[4]
num_samp = as.numeric(args[5])
ncores = as.numeric(args[6])

print("in_dir:")
print(in_dir)

print("out_dir:")
print(out_dir)

print("file_id_train:")
print(file_id_train)

print("file_id_test:")
print(file_id_test)

print("num_samp:")
print(num_samp)

print("ncores:")
print(ncores)

run_bayes_prism(in_dir, out_dir, file_id_train, file_id_test, num_samp, ncores)
