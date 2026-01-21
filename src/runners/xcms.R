suppressMessages(library(MSnbase))
suppressMessages(library(xcms))
suppressMessages(library(optparse))
option_list <- list(
  make_option(c("-d", "--dir"), type="character", help="mzML 文件夹路径"),
  make_option(c("-o", "--output"), type="character", help="输出文件路径"),
  make_option(c("--polarity"), type="character", default="positive"),
  make_option(c("--mz_tol"), type="numeric", default=15),    # 针对 Q-TOF 的 ppm
  make_option(c("--minwidth"), type="numeric", default=10),
  make_option(c("--maxwidth"), type="numeric", default=60),
  make_option(c("--noise"), type="numeric", default=1000),
  make_option(c("--sn"), type="numeric", default=10),
  make_option(c("--prefilter"), type="numeric", default=3),
  make_option(c("--frac"), type="numeric", default=0.5)
)
opt <- parse_args(OptionParser(option_list=option_list))
cat(sprintf("\n[R] 正在读取目录: %s\n", opt$dir))
files <- list.files(opt$dir, pattern = "\\.mzML$", full.names = TRUE, ignore.case = TRUE)
raw_data = readMSData(files,mode="onDisk",msLevel=1)
raw_data <- filterEmptySpectra(raw_data)
params <- CentWaveParam(
    ppm = opt$mz_tol, 
    peakwidth = c(opt$minwidth, opt$maxwidth),
    noise = opt$noise,
    snthresh = opt$sn,
    prefilter = c(opt$prefilter, 300), # 100 为强度阈值，针对 Agilent 仪器优化
    mzCenterFun = "wMean",
    integrate = 1,
    fitgauss = FALSE
)
xdata <- findChromPeaks(raw_data, params, return.type="XCMSnExp")

pdp <- PeakDensityParam(sampleGroups = rep(1, length(files)), minFraction = opt$frac, bw = 5)

xdata <- groupChromPeaks(xdata, param = pdp)
xdata <- adjustRtime(xdata, ObiwarpParam(binSize = 1,))
xdata <- groupChromPeaks(xdata, param = pdp) # 校正后二次分组
xdata <- fillChromPeaks(xdata)               # 填充缺失峰，提高定量准确度

peakInfo <- featureDefinitions(xdata)
intensities <- featureValues(xdata, value = "into") # 使用面积(into)而非峰高
peaktable <- merge(peakInfo, intensities, by = 'row.names', all = TRUE)
drop_cols <- c("peakidx", "Row.names")
peaktable <- peaktable[, !(names(peaktable) %in% drop_cols)]
write.table(peaktable, file = opt$output, sep="\t", row.names=FALSE, col.names=TRUE, quote=FALSE)