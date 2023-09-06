The command line to upscaled an image will be:
FidelityFX_CLI.exe [Options] <SrcFile1> <DstFile1> <SrcFile2> <DstFile2> ...
Options:
-Scale <DstWidth> <DstHeight>
	Width, Height can be:
	Number: -Scale 2340 1080
	Scale factor: -Scale 1.5x 1.5x
-Mode <Mode>
	Mode can be:
	-Mode RASU (SGSR)
	-Mode Linear
	-Mode NearestNeighbor
Eg:
FidelityFX_CLI.exe -Mode RASU -Scale 2340 1080 Discovery2_AA_1552x720.png Discovery2_SGSR_VR_1552x720_to_2340x1080.png
FidelityFX_CLI.exe -Mode Linear -Scale 2340 1080 Discovery2_AA_1552x720.png Discovery2_Linear_1552x720_to_2340x1080.png
FidelityFX_CLI.exe -Mode RASU -Scale 1.5x 1.5x Discovery2_AA_1552x720.png Discovery2_SGSR_VR_1552x720_to_2328x1080.png
FidelityFX_CLI.exe -Mode RASU -Scale 1.5x 1.5x Discovery2_AA_1552x720.png Discovery2_SGSR_VR_1552x720_to_2328x1080.png
FidelityFX_CLI.exe -Mode RASU -Scale 2x 2x