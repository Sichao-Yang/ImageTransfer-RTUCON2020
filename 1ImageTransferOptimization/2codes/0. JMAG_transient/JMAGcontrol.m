%% this script is used to launch JMAG, do the calculation and export the data

root = 'E:\2. Work Project\FEM_ImageTransfer\200423-optimizationWithDL\00. tools & codes\0. JMAG FEA GA\used FE model\';
% Launch JMAG-Designer
% Select JMAG-Designer version and start JMAG-Designer
designer = actxserver('designer.Application.181');
% Display JMAG-Designer window
designer.Show();
app = designer;
app.Load(strcat(root,'Geo.jproj'))
app.GetModel(0).GetStudy(studyNo).GetDesignTable().Import(strcat(root,'input.csv'))
% remove the old caseno. 0
app.GetModel(0).GetStudy(studyNo).GetDesignTable().RemoveCase(0)
app.GetModel(0).RestoreCadLinkWithFilePath(strcat(root,'Geo.jfiles/assembly/Geo - Copy_0.jmd'))
app.GetModel(0).GetStudy(studyNo).ApplyAllCasesCadParameters()
% run all cases
app.GetModel(0).GetStudy(studyNo).RunAllCases()
% export all results
app.GetModel(0).GetStudy(studyNo).ExportCaseValueData(strcat(root,'FEout.csv'))

% remove alll results BY force open new project without save
app.NewProject("Untitled")
app.Load(strcat(root,'Geo.jproj'))