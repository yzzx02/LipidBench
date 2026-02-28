suppressWarnings(suppressMessages(library(MSnbase)))
suppressWarnings(suppressMessages(library(xcms)))
suppressMessages(library(optparse))
suppressMessages(library(BiocParallel))

option_list <- list(
  make_option(c("-d", "--dir"), type = "character", help = "mzML 文件夹路径"),
  make_option(c("-o", "--output"), type = "character", help = "输出文件路径"),
  make_option(c("--polarity"), type = "character", default = "positive"),
  make_option(c("--mz_tol"), type = "numeric", default = 15),
  make_option(c("--minwidth"), type = "numeric", default = 10),
  make_option(c("--maxwidth"), type = "numeric", default = 60),
  make_option(c("--noise"), type = "numeric", default = 1000),
  make_option(c("--sn"), type = "numeric", default = 10),
  make_option(c("--prefilter"), type = "numeric", default = 3),
  make_option(c("--frac"), type = "numeric", default = 0.5),
  make_option(c("--mzdiff"), type = "numeric", default = 0.001)
)
register(SerialParam(progressbar = TRUE))
opt <- parse_args(OptionParser(option_list = option_list))

files <- list.files(opt$dir, pattern = "\\.mzML$", full.names = TRUE, ignore.case = TRUE)
raw_data = readMSData(files,mode="onDisk",msLevel=1)
raw_data <- filterEmptySpectra(raw_data)
params <- CentWaveParam(
    ppm = opt$mz_tol,
    peakwidth = c(opt$minwidth, opt$maxwidth),
    noise = opt$noise,
    snthresh = opt$sn,
    mzdiff = opt$mzdiff,
    prefilter = c(opt$prefilter, 1000),
    mzCenterFun = "wMean",
    integrate = 1,
    fitgauss = FALSE
)
xdata <- findChromPeaks(raw_data, params, return.type="XCMSnExp")

pdp <- PeakDensityParam(sampleGroups = rep(1, length(files)), minFraction = opt$frac, bw = 10,binSize = 0.005)
xdata <- groupChromPeaks(xdata, param = pdp)

if (length(files) > 1) {
    xdata <- adjustRtime(xdata, ObiwarpParam(binSize = 1))
    xdata <- groupChromPeaks(xdata, param = pdp)
    xdata <- fillChromPeaks(xdata)
}

peakInfo <- featureDefinitions(xdata)
if (length(files) == 1) {
    pks <- chromPeaks(xdata)
    peak_indices <- sapply(peakInfo$peakidx, function(x) x[1])
    peakInfo$mzmin <- pks[peak_indices, "mzmin"]
    peakInfo$mzmax <- pks[peak_indices, "mzmax"]
    peakInfo$rtmin <- pks[peak_indices, "rtmin"]
    peakInfo$rtmax <- pks[peak_indices, "rtmax"]
}

intensities <- featureValues(xdata, value = "into")
snrs <- featureValues(xdata, value = "sn")
peaktable <- merge(peakInfo, intensities, by = 'row.names' , all = TRUE)

is_single_sample <- is.null(dim(snrs)) || (length(dim(snrs)) == 2 && ncol(snrs) == 1)
if (is_single_sample) {
    snrs_df <- as.data.frame(snrs)
    colnames(snrs_df) <- "sn"
    peaktable <- merge(peaktable, snrs_df, by.x = "Row.names", by.y = "row.names", all.x = TRUE)
}

drop_cols <- c("peakidx")
peaktable <- peaktable[, !(names(peaktable) %in% drop_cols)]
write.table(peaktable, file = opt$output, sep=",", row.names= FALSE, col.names=TRUE, quote=FALSE)
