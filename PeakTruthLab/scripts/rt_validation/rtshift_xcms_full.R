suppressPackageStartupMessages(library(MSnbase))
suppressPackageStartupMessages(library(xcms))
suppressPackageStartupMessages(library(BiocParallel))
register(SerialParam(progressbar=FALSE))
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==4L)
input_file <- args[1]; table_file <- args[2]; output_file <- args[3]; sample_id <- args[4]
targets <- read.csv(table_file,stringsAsFactors=FALSE)
raw <- readMSData(input_file,mode='onDisk',msLevel=1L)
raw <- filterEmptySpectra(raw)
param <- CentWaveParam(ppm=5,peakwidth=c(5,50),noise=3000,snthresh=5,
                      prefilter=c(3,3000),mzdiff=0.001,mzCenterFun='wMean',
                      integrate=1,fitgauss=FALSE)
detected <- findChromPeaks(raw,param,return.type='XCMSnExp')
pk <- chromPeaks(detected)
write.csv(pk,sub('[.]csv$','_all_chromPeaks.csv',output_file),row.names=TRUE)
rows <- vector('list',nrow(targets)); candidates <- list()
for(i in seq_len(nrow(targets))) {
    idx <- which(abs(pk[,'mz']-targets$reference_mz[i])<=targets$reference_mz[i]*10e-6 &
                 abs(pk[,'rt']/60-targets$reference_rt[i])<=1 &
                 is.finite(pk[,'into']) & pk[,'into']>0)
    if(length(idx)) {
        all <- data.frame(candidate_index=idx,pred_apex_rt=pk[idx,'rt']/60,
            pred_left=pk[idx,'rtmin']/60,pred_right=pk[idx,'rtmax']/60,
            pred_area=pk[idx,'into'],pred_score=pk[idx,'sn'],pred_mz=pk[idx,'mz'])
        all$feature_id <- targets$feature_id[i]; all$sample <- sample_id
        candidates[[length(candidates)+1L]] <- all
        ord <- order(round(abs(all$pred_apex_rt-targets$reference_rt[i]),3),-all$pred_score)
        best <- all[ord[1],]
        rows[[i]] <- data.frame(feature_id=targets$feature_id[i],sample=sample_id,pred_found=1L,
            valid_candidate_count=nrow(all),pred_apex_rt=best$pred_apex_rt,pred_left=best$pred_left,
            pred_right=best$pred_right,pred_area=best$pred_area,pred_score=best$pred_score)
    } else rows[[i]] <- data.frame(feature_id=targets$feature_id[i],sample=sample_id,pred_found=0L,
         valid_candidate_count=0L,pred_apex_rt=NA_real_,pred_left=NA_real_,pred_right=NA_real_,pred_area=NA_real_,pred_score=NA_real_)
}
write.csv(do.call(rbind,rows),output_file,row.names=FALSE)
if(length(candidates)) write.csv(do.call(rbind,candidates),sub('[.]csv$','_candidates.csv',output_file),row.names=FALSE)
cat('xcms=',as.character(packageVersion('xcms')),' sample=',sample_id,' targets=',nrow(targets),
    ' found=',sum(vapply(rows,function(x)x$pred_found,integer(1))),'\n',sep='')
