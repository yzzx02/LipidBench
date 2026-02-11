suppressWarnings(suppressMessages(library(MSnbase)))
suppressWarnings(suppressMessages(library(xcms)))
suppressMessages(library(optparse))
suppressMessages(library(BiocParallel))

# Force serial execution with progress bar to show activity
# Serial is used to avoid Windows-specific parallel processing environment issues detected previously

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
register(SerialParam(progressbar = TRUE)) # Add progressbar back for feedback
opt <- parse_args(OptionParser(option_list = option_list))
cat(sprintf("\n[R] 正在读取目录: %s\n", opt$dir))
files <- list.files(opt$dir, pattern = "\\.mzML$", full.names = TRUE, ignore.case = TRUE) # nolint
raw_data = readMSData(files,mode="onDisk",msLevel=1) # nolint
raw_data <- filterEmptySpectra(raw_data)
params <- CentWaveParam(
    ppm = opt$mz_tol,  # nolint
    peakwidth = c(opt$minwidth, opt$maxwidth),
    noise = opt$noise,
    snthresh = opt$sn,
    mzdiff = opt$mzdiff,
    prefilter = c(opt$prefilter, 3000), # 使用噪声水平作为强度阈值，避免生成过多ROI
    mzCenterFun = "wMean",
    integrate = 1,
    fitgauss = FALSE
)
xdata <- findChromPeaks(raw_data, params, return.type="XCMSnExp") # nolint

pdp <- PeakDensityParam(sampleGroups = rep(1, length(files)), minFraction = opt$frac, bw = 10,binSize = 0.005)
xdata <- groupChromPeaks(xdata, param = pdp)

if (length(files) > 1) {
    cat("[R] 检测到多个文件，正在进行保留时间校正 (obiwarp)...\n")
    xdata <- adjustRtime(xdata, ObiwarpParam(binSize = 1))
    xdata <- groupChromPeaks(xdata, param = pdp) # 校正后二次分组
    xdata <- fillChromPeaks(xdata)               # 填充缺失峰，提高定量准确度
} else {
    cat("[R] 仅检测到单个文件，跳过保留时间校正。\n")
}

peakInfo <- featureDefinitions(xdata)

# 当只有一个文件时，featureDefinitions 默认计算的是 Peak 中心点的范围（即只有一个点）
# 因此 mzmin/mzmax 会等于 mzmed。这里手动替换为 Peak 的实际边界范围。
if (length(files) == 1) {
    pks <- chromPeaks(xdata)
    peak_indices <- sapply(peakInfo$peakidx, function(x) x[1])
    peakInfo$mzmin <- pks[peak_indices, "mzmin"]
    peakInfo$mzmax <- pks[peak_indices, "mzmax"]
    peakInfo$rtmin <- pks[peak_indices, "rtmin"]
    peakInfo$rtmax <- pks[peak_indices, "rtmax"]
}

intensities <- featureValues(xdata, value = "into") # 使用面积(into)而非峰高
snrs <- featureValues(xdata, value = "sn") # 提取信噪比

peaktable <- merge(peakInfo, intensities, by = 'row.names' , all = TRUE)

# 仅当单样本时加入 sn 列
# 如果 snrs 是向量或者单列矩阵，则认为是单样本
is_single_sample <- is.null(dim(snrs)) || (length(dim(snrs)) == 2 && ncol(snrs) == 1)

if (is_single_sample) {
    snrs_df <- as.data.frame(snrs)
    colnames(snrs_df) <- "sn"
    peaktable <- merge(peaktable, snrs_df, by.x = "Row.names", by.y = "row.names", all.x = TRUE)
}

drop_cols <- c("peakidx")
peaktable <- peaktable[, !(names(peaktable) %in% drop_cols)]
write.table(peaktable, file = opt$output, sep=",", row.names= FALSE, col.names=TRUE, quote=FALSE)