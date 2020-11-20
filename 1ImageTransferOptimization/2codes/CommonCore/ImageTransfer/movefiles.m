source=strcat(root,'checkpoint\ITimage\*.jpg');
destination=strcat(root,'checkpoint\ITimage\gen', num2str(gen));
movefile(source, destination)

source=strcat(root,'checkpoint\ITtransfered\*.jpg');
destination=strcat(root,'checkpoint\ITtransfered\gen', num2str(gen));
movefile(source, destination)