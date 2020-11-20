%*************************************************************************
% This is used to control JMAG: launch, calculate and export the data
% 	01-Jun-2020 sichao yang
%*************************************************************************
function JMAGcontrol_v2(Switch, inputfile, datafile, root) 
% Launch JMAG-Designer
% Select JMAG-Designer version and start JMAG-Designer
designer = actxserver('designer.Application.181');
% Display JMAG-Designer window
designer.Show();
app = designer;
studyNo = 0;    % the study no. in the model project
if Switch == 0
    modelname='2. controled FE model\Opt_VPM_stat';
elseif Switch ==1
    modelname='2. controled FE model\IT_VPM';
elseif Switch ==3
    modelname='2. controled FE model\Opt_SPM';
end
app.Load(strcat(root,modelname,'.jproj'))
app.GetModel(0).GetStudy(studyNo).GetDesignTable().Import(inputfile)
% remove the old caseno. 0
app.GetModel(0).GetStudy(studyNo).GetDesignTable().RemoveCase(0)
app.GetModel(0).RestoreCadLinkWithFilePath(strcat(root,modelname,'.jfiles','/assembly/Geo - Copy_0.jmd'))
app.GetModel(0).GetStudy(studyNo).ApplyAllCasesCadParameters()
% run all cases
app.GetModel(0).GetStudy(studyNo).RunAllCases()
% export all results
app.GetModel(0).GetStudy(studyNo).ExportCaseValueData(datafile)

% remove alll results BY force open new project without saving
app.NewProject("Untitled")
app.Load(strcat(root,modelname,'.jproj'))