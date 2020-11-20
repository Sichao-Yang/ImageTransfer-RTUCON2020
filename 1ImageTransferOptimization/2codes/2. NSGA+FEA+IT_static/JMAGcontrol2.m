%*************************************************************************
% This is used to control JMAG: launch, calculate and export the data
% 	01-Jun-2020 sichao yang
%*************************************************************************
function JMAGcontrol(Switch,root)
%% this script is used to launch JMAG, do the calculation and export the data
if Switch == 0
    studyNo = 1;  
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

elseif Switch ==1
    studyNo = 0;    % this study is used for image extraction, its mesh is coarse for speed
    % Launch JMAG-Designer
    % Select JMAG-Designer version and start JMAG-Designer
    designer = actxserver('designer.Application.181');
    % Display JMAG-Designer window
    designer.Show();
    app = designer;
    app.Load(strcat(root,'IT.jproj'))
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
    app.Load(strcat(root,'IT.jproj'))
end