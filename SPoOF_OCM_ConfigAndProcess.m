%%

% clear all;
OCMIntMap = OCMMap_viridis(256);
OCMPhaseMap = circshift(SPoOFMap_colorcet('C3'),128);
OCMDivMap = SPoOFMap_colorcet('D5');

%%
Ftoshow = 1 ;

FcxOff = [0 -13 0 0  0 0 0 0];
FcyOff = [0  0  0  2 0 0 0 0];


% fbase = 'Dish3_7200fps_512x512_10uMGlut_C001H001S0001';
fbase = FDir(Ftoshow).name;
bbase = 'INVALID';
OutName1 = 'Processed_P1';
OutName2 = 'Processed_P2';
numX = 768;
numY = 768;

numF = floor(dir([pwd '\' fbase '\' fbase '.mraw']).bytes/(numX*numY*1.5));

% numF = 43000;

numx = 320;
numy = 320;

FCx1 = 18 - FcxOff(Ftoshow);
FCy1 = 89 - FcyOff(Ftoshow);

Rad = 160;

FCx2 = 50 + FcxOff(Ftoshow);
FCy2 = 362 + FcyOff(Ftoshow);

[X, Y] = meshgrid( linspace(-1,1,numx),linspace(-1,1,numy));
RHO = sqrt(X.^2  + Y.^2);
Win = (RHO.^2 <=2);

FID = fopen('Config.txt','wt');
fprintf(FID,'%s\n',[pwd '\' fbase '\' fbase '.mraw']);
fprintf(FID,'%s\n','INVALID');
% fprintf(FID,'%s\n',[pwd '\' bbase '\' bbase '.mraw']);
fprintf(FID,'%s\n',[pwd '\' fbase '\' OutName1 '.bin']);
fprintf(FID,'%s\n',[pwd '\' fbase '\' OutName2 '.bin']);
fprintf(FID,'%s\n',[pwd '\' fbase '\RawOut.bin']);
fprintf(FID,'%d %d %d %d %d\n',numX, numY, numx, numy, numF);
fprintf(FID,'%d %d %f\n',FCx1, FCy1, Rad);
fprintf(FID,'%d %d %f\n',FCx2, FCy2, Rad);
fclose(FID);
%
system(['D:\Rishee\SPoOFOCM_Proc\SPoOFOCM_PostProc\x64\Debug\SPoOFOCM_PostProc.exe ' pwd '\Config.txt']);

%
FSkip = 3;
FID1 = fopen([pwd '\' fbase '\' OutName1 '.bin'],'rb');
fseek(FID1,numx*numy*2*FSkip*4,'bof');
Out_P1 = fread(FID1,numx*numy*2,'float');
Out_P1 = reshape( Out_P1(1:2:end-1) + 1i.*Out_P1(2:2:end),[numx numy]);
fclose(FID1);

FID2 = fopen([pwd '\' fbase '\' OutName2 '.bin'],'rb');
fseek(FID2,numx*numy*2*FSkip*4,'bof');
Out_P2 = fread(FID2,numx*numy*2,'float');
Out_P2 = reshape( Out_P2(1:2:end-1) + 1i.*Out_P2(2:2:end),[numx numy]);
fclose(FID2);
%

% DFWin = 2;
% Out_P1_DF = fftshift(fft2(Out_P1))/(160000);
% Out_P1_DF(end/2-DFWin+1:end/2+DFWin,end/2-DFWin+1:end/2+DFWin) = 0;
% Out_P1_DF = (fft2(ifftshift(Out_P1_DF)));
% 
% Out_P2_DF = fftshift(fft2(Out_P2))/(160000);
% Out_P2_DF(end/2-DFWin+1:end/2+DFWin,end/2-DFWin+1:end/2+DFWin) = 0;
% Out_P2_DF = (fft2(ifftshift(Out_P2_DF)));

figure(3);
clf;
set(gcf,'Color',[1 1 1]);
subplot(321);
hold all;
imagesc( abs( Out_P1'/3).^0.8 .* Win,'AlphaData',Win);
colormap(gca,OCMIntMap);
axis image;
% colorbar;
caxis([0 1]*500)
set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI');
title('|E_{Vs}|','Color',[1 1 1]*0);
colorbar('Location','EastOutside','LineWidth',1.5);


subplot(322);
hold all;
cla;
imagesc( abs( Out_P2'/2).^0.8 .* Win,'AlphaData',Win);
colormap(gca,OCMIntMap);
caxis([0 1]*500)
set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI');
title('|E_{Hs}|','Color',[1 1 1]*0);
axis image;
colorbar('Location','EastOutside','LineWidth',1.5);


subplot(323);
imagesc( Win.*-angle(Out_P1)','AlphaData', Win);
colormap(gca,OCMPhaseMap);
axis image;
set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI');
title('\angleE_{Vs}','Color',[1 1 1]*0);
caxis([-pi pi]);
c = colorbar('Location','EastOutside','LineWidth',1.5,'Ticks',[-pi -pi/2 0 pi/2 pi],'TickLabels',{'-π','-π/2','0','π/2','π'});

subplot(324);
imagesc( Win.* angle(Out_P2)','AlphaData', Win);
colormap(gca,OCMPhaseMap);
axis image;
set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI');
title('\angleE_{Hs}','Color',[1 1 1]*0);
caxis([-pi pi]);
c = colorbar('Location','EastOutside','LineWidth',1.5,'Ticks',[-pi -pi/2 0 pi/2 pi],'TickLabels',{'-π','-π/2','0','π/2','π'});

subplot(325);
imagesc( [log(abs( fftshift( fft2( Out_P1)))) log(abs( fftshift( fft2( Out_P2))))]);
colormap(gca,jet);
axis image;
set(gca,'Color','k');
set(gca,'XTick',[],'YTick',[],'FontSize',15);
colorbar('Location','EastOutside','LineWidth',1.5);

% caxis([30 80]);

% subplot(326);
% imagesc( log(abs( fftshift( fft2( Out_P2)))));
% colormap(gca,jet);
% axis image;
% set(gca,'Color','k');
% set(gca,'XTick',[],'YTick',[],'FontSize',15);
% colorbar('Location','EastOutside','LineWidth',1.5);
% % caxis([30 80]);

subplot(326);
imagesc( Win.* angle(Out_P2.*Out_P1)','AlphaData', Win);
colormap(gca,OCMPhaseMap);
axis image;
set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI');
title('\angleE_{Hs}','Color',[1 1 1]*0);
caxis([-pi pi]);
c = colorbar('Location','EastOutside','LineWidth',1.5,'Ticks',[-pi -pi/2 0 pi/2 pi],'TickLabels',{'-π','-π/2','0','π/2','π'});

%%