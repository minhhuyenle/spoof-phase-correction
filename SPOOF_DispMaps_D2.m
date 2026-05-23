clear all;
OCMPhaseMap = circshift(SPoOFMap_colorcet('C3'),0);

%%
FDir = dir('*768x768*');

 fbase = FDir(1).name;
    disp(fbase);
    
 %%   
bbase = 'INVALID';
OutName1 = 'Processed_P1';
OutName2 = 'Processed_P2';
numX = 768;
numY = 768;

numF = floor(dir([pwd '\' fbase '\' fbase '.mraw']).bytes/(numX*numY*1.5));

% numF = 43000;

numx = 320;
numy = 320;

Sigma = 0.04;
[X, Y] = meshgrid( linspace(-1,1,numx),linspace(-1,1,numy));
RHO = sqrt(X.^2  + Y.^2);
G2D = exp( - ( RHO.*RHO)/(2.*Sigma.^2));
G2D = G2D / max(G2D(:));
G2D = 1 - G2D;
Win = (RHO.^2 <=2);

y1 = -4;
y2 = 35;
x1 = 0;
x2 = 30;
xyc = -1.5;
ycen = 0;
xcen = 0;
off = 0;
[x,y] = meshgrid( linspace(-1,1,numx),linspace(-1,1,numy));
PhaseMask1 = y1.*y + x1.*x + y2.*(y-ycen).^2 + x2.*(x-xcen).^2 + xyc.*x.*y + off;

y1 = 2.0;
y2 = -34;
x1 = 1.3;
x2 = -30;
xyc = -4;
ycen = 0;
xcen = 0;
off = 0.3;
[x,y] = meshgrid( linspace(-1,1,numx),linspace(-1,1,numy));
PhaseMask2 = y1.*y + x1.*x + y2.*(y-ycen).^2 + x2.*(x-xcen).^2 + xyc.*x.*y + off;

fs = 1000;
taxis = (0:(numF-1))/fs;

%
figure(300);
set(gcf,'Color',[0 0 0],'InvertHardCopy','off');

FID1 = fopen([pwd '\' fbase '\' OutName1 '.bin'],'rb');
FID2 = fopen([pwd '\' fbase '\' OutName2 '.bin'],'rb');

FStart = 10000;

fseek(FID1,numx*numy*2*4*FStart,'bof');
fseek(FID2,numx*numy*2*4*FStart,'bof');

numFOneBeat = 8000;

Back1x = 45:64;%1:numx;
Back1y = 125:144;%1:numy;

FAvg = 5;
numFrames = numF;

yShow = 21:numy;
xShow = 1:(numx-20);

PhaseColl = zeros([numx numy (numFOneBeat/FAvg)]);
RetColl = zeros([numx numy (numFOneBeat/FAvg)]);
DispFColl = zeros([numx numy 2 (numFOneBeat/FAvg)]);
DispGColl = zeros([numx numy 2 (numFOneBeat/FAvg)]);
    
v = VideoWriter('HeartBeat_DispMaps.mp4','MPEG-4');
v.FrameRate = 50;
v.Quality = 80;

open(v)

for frameIdx = 1%1:(numFOneBeat/FAvg)
    Out_P1 = fread(FID1,numx*numy*2 * FAvg,'float');
    Out_P1 = reshape( Out_P1(1:2:end-1) + 1i.*Out_P1(2:2:end),[numx numy FAvg]);

    Out_P2 = fread(FID2,numx*numy*2*FAvg,'float');
    Out_P2 = reshape( Out_P2(1:2:end-1) + 1i.*Out_P2(2:2:end),[numx numy FAvg]);

    Out_P1 = exp(-1i*PhaseMask1).*Out_P1;
    Out_P2 = exp(1i*PhaseMask2).*conj(Out_P2).*exp(1i.*-pi/2);
    
    angP1 = median( angle( Out_P1(:,:,round(FAvg/2))),3);
    BackSub = median( reshape(angP1(Back1x,Back1y),[length(Back1x)*length(Back1y) 1]));
    angP1 = angle( exp(1i.*angP1).*exp(-1i.*BackSub).*exp(1i.*pi/5));
    angP2 = median( angle( Out_P2(:,:,round(FAvg/2))),3);
    angP2 = angle( exp(1i.*angP2).*exp(-1i.*BackSub).*exp(1i.*pi/1.5));

    delta_s = 180/pi*atan2( abs( 1i.*Out_P2 - Out_P1), abs( 1i.*Out_P2 + Out_P1));

%%
    if frameIdx == 1
        FirstAng = angP1;
        FirstDel = delta_s(:,:,round(FAvg/2));
        angP1_Prev = angP2;
        delta_s_Prev = delta_s(:,:,round(FAvg/2));
    end

    [Xu,Yu] = meshgrid( (1:numx),(1:numy));

    [DispF] = 2*imregdemons(angP1,FirstAng);
    DispF(:,:,1) = medfilt2(DispF(:,:,1),[5 5]);
    DispF(:,:,2) = medfilt2(DispF(:,:,2),[5 5]);

    [DispG] = 2*imregdemons(delta_s(:,:,round(FAvg/2)),FirstDel);
    DispG(:,:,1) = medfilt2(DispG(:,:,1),[5 5]);
    DispG(:,:,2) = medfilt2(DispG(:,:,2),[5 5]);

    DispFCond = repmat(sqrt(DispF(:,:,1).^2 + DispF(:,:,2).^2),[1 1 2]);
    DispGCond = repmat(sqrt(DispG(:,:,1).^2 + DispG(:,:,2).^2),[1 1 2]);
    DispF(DispFCond < 3) = NaN;
    DispG(DispGCond < 3) = NaN;  

    DispFColl(:,:,:,frameIdx) = DispF;
    DispGColl(:,:,:,frameIdx) = DispG;
    PhaseColl(:,:,frameIdx) = angP1;
    RetColl(:,:,frameIdx) = medfilt2(mean( delta_s(:,:,round(FAvg/2)),3),[3 3]);

    clf;
    ArrowGap = 7;

    subplot(2,2,1);
    imagesc(angP1);
    text(10,12,[num2str(taxis((frameIdx-1)*FAvg + 1 + FStart),'%1.3f') ' s'],'FontName','Segoe UI','FontSize',20,'Color','y','FontWeight','bold');
    axis image;
    set(gca,'XColor','k','YColor','k','XTick',[],'YTick',[],'FontName','Segoe UI','FontSize',15,'YDir','reverse');
    colormap(gca,OCMPhaseMap);
    clim([-pi pi]);
    ylabel('OCM phase (rad)','Color','w');
    c = colorbar('Location','EastOutside','LineWidth',1.5,'Color','w');
    ylim([min(yShow) max(yShow)]);
    xlim([min(xShow) max(xShow)]);

    subplot(2,2,2); hold all;
%     imagesc(sqrt(DispF(:,:,1).^2 + DispF(:,:,2).^2));
    imagesc(atan2(DispF(:,:,2),DispF(:,:,1))*180/pi,'AlphaData',~isnan(mean(DispF,3)));
    quiver(Xu(3:ArrowGap:end,3:ArrowGap:end),Yu(3:ArrowGap:end,3:ArrowGap:end),...
        2* DispF(3:ArrowGap:end,3:ArrowGap:end,1),...
        2* DispF(3:ArrowGap:end,3:ArrowGap:end,2),0,'w','LineWidth',0.25)
    axis image;
    set(gca,'XColor','w','YColor','w','XTick',[],'YTick',[],'FontName','Segoe UI','FontSize',15,'Color','k','YDir','reverse');
    colormap(gca,SPoOFMap_colorcet('C1'));
    ylim([min(yShow) max(yShow)]);
    xlim([min(xShow) max(xShow)]);
    clim([-pi pi]*180/pi);
    box on;
    c = colorbar('Location','EastOutside','LineWidth',1.5,'Color','w');
    c.Label.String = {'Displacement direction ','of phase (^o)'};
    
    subplot(2,2,3);
    imagesc(medfilt2(mean( delta_s(yShow,xShow,round(FAvg/2)),3),[3 3]));
%         text(20,20,[num2str(taxis(frameIdx),'%1.3f') ' s'],'FontName','Segoe UI','FontSize',15,'Color','y');
    axis image;
    set(gca,'XColor','k','YColor','k','XTick',[],'YTick',[],'FontName','Segoe UI','FontSize',15,'YDir','reverse');
    colormap(gca,SPoOFMap_colorcet('R2'));
    c = colorbar('Location','EastOutside','LineWidth',1.5,'Color','w');
    clim([0 50]);
    ylabel('Retardation angle (^o)','Color','w');

    subplot(2,2,4); hold all;
    imagesc(atan2(DispG(:,:,2),DispG(:,:,1))*180/pi,'AlphaData',~isnan(mean(DispG,3)));
    quiver(Xu(3:ArrowGap:end,3:ArrowGap:end),Yu(3:ArrowGap:end,3:ArrowGap:end),2*DispG(3:ArrowGap:end,3:ArrowGap:end,1), 2*DispG(3:ArrowGap:end,3:ArrowGap:end,2),0,'w','LineWidth',0.25)
    axis image;
    set(gca,'XColor','w','YColor','w','XTick',[],'YTick',[],'FontName','Segoe UI','FontSize',15,'Color','k','YDir','reverse');
    ylim([min(yShow) max(yShow)]);
    xlim([min(xShow) max(xShow)]);
    colormap(gca,SPoOFMap_colorcet('C1'));
    c = colorbar('Location','EastOutside','LineWidth',1.5,'Color','w');
    c.Label.String = {'Displacement direction ','of retardation angle (^o)'};
    clim([-pi pi]*180/pi);
    box on;

%%

    angP1_Prev = angP1;
    delta_s_Prev = delta_s(:,:,round(FAvg/2));

    FrameCurr = getframe(gcf);
    writeVideo(v,FrameCurr);
end

fclose(FID2);
fclose(FID1);
close(v)
%%
save('DispmapSave.mat','FStart','numFOneBeat','Back1x','Back1y','FAvg','PhaseColl','RetColl','DispFColl','DispGColl','-v7.3');

%%

figure(302);
set(gcf,'Color',[1 1 1]);
DispAbs = squeeze( sqrt( DispFColl(:,:,1,:).^2 +  DispFColl(:,:,2,:).^2 ));
DispAbs(:,1:min(xShow)) = NaN;
DispAbs(1:min(yShow),:) = NaN;
DispAbs(:,max(xShow):numy) = NaN;
DispAbs(max(yShow):numx,:) = NaN;
DispAbs(DispAbs > 10) = NaN;

DispPhase = squeeze( atan2(DispFColl(:,:,1,:),DispFColl(:,:,2,:)) );
DispPhase(:,1:min(xShow)) = NaN;
DispPhase(1:min(yShow),:) = NaN;
DispPhase(:,max(xShow):numy) = NaN;
DispPhase(max(yShow):numx,:) = NaN;

MeanDisp = sum( reshape(DispAbs,numx*numy,[]),1,'omitnan')./(numx*numy);
MeanPhase = median( reshape(DispPhase,numx*numy,[]),1,'omitnan');
CentX =  sum(squeeze(sum(DispAbs,1,'omitnan')).*repmat( (1:numx)',[1 length(MeanDisp)]),'omitnan')./ ...
    sum(squeeze(sum(DispAbs,1,'omitnan')),'omitnan');
CentY =  sum(squeeze(sum(DispAbs,2,'omitnan')).*repmat( (1:numy)',[1 length(MeanDisp)]),'omitnan')./ ...
    sum(squeeze(sum(DispAbs,2,'omitnan')),'omitnan');
% [~,CentX] =max( sum(DispAbs,2,'omitnan'));
% [~,CentY] = max( sum(DispAbs,1,'omitnan'));

CentX(MeanDisp < 0.05) = NaN;
CentY(MeanDisp < 0.05) = NaN;

CentX = movmean(squeeze(CentX),4);
CentY = movmean(squeeze(CentY),4);

FShow = 900:1600;%1100:1800;%

TimeCols = SPoOFMap_colorcet('R4','N',ceil(length(FShow)/5)+4);

figure(302);
set(gcf,'Color',[1 1 1]);
clf;

subplot(2,2,1); hold all;
imagesc(PhaseColl(:,:,1),'AlphaData',0.5)
for idx = FShow
    scatter(squeeze(CentX(idx)),squeeze(CentY(idx)),...
    20,taxis(idx*FAvg + FStart),'filled','MarkerFaceAlpha',0.8,'MarkerEdgeColor','none','MarkerFaceColor',TimeCols(round((idx - min(FShow))/5) + 1,:));

    plot(CentX(idx)+[0 -4*(MeanDisp(idx)+2)*sin(MeanPhase(idx))],CentY(idx)+[0 -4*(MeanDisp(idx)+2)*cos(MeanPhase(idx))],...
        'Color',[TimeCols(round((idx - min(FShow))/5) + 1,:) 0.8]);
end
set(gca,'XColor','k','YColor','k','FontName','Segoe UI','FontSize',15,'LineWidth',1.5,'YDir','reverse','XTick',[],'YTick',[],'Color','w');
colormap(gca,SPoOFMap_colorcet('C5'));
% colormap(gca,TimeCols);
% c = colorbar('Location','EastOutside','LineWidth',1.5,'Color','k');
% c.Label.String = 'Time (s)';
box on;
axis image;
ylim([min(yShow) max(yShow)]);
xlim([min(xShow) max(xShow)]);
clim([-pi pi])
% clim([min(taxis(FShow*FAvg + FStart)) max(taxis(FShow*FAvg + FStart))]);
% xlabel(('x (px)'))
% ylabel(('y (px)'))
title({'Displacement vectors at their ','centroids along time'})
%
subplot(4,2,5); hold all;
scatter(taxis(FShow*FAvg + FStart),squeeze(CentY(FShow)),...
    20,taxis(FShow*FAvg + FStart),'filled');
set(gca,'XColor','k','YColor','k','FontName','Segoe UI','FontSize',15,'LineWidth',1.5,'YDir','normal');
box on;
colormap(gca,TimeCols);
clim([min(taxis(FShow*FAvg + FStart)) max(taxis(FShow*FAvg + FStart))]);
xlim([min(taxis(FShow*FAvg + FStart)) max(taxis(FShow*FAvg + FStart))]);
ylabel({'Centroid along','y (px)'})
% xlabel(('Time (s)'))
set(gca,'Position',get(gca,'Position') - [-0.02 0 0.05 0]);
ylim([40 200]);

subplot(4,2,7); hold all;
scatter(taxis(FShow*FAvg + FStart),squeeze(CentX(FShow)),...
    20,taxis(FShow*FAvg + FStart),'filled');
set(gca,'XColor','k','YColor','k','FontName','Segoe UI','FontSize',15,'LineWidth',1.5,'YDir','normal');
box on;
colormap(gca,TimeCols);
clim([min(taxis(FShow*FAvg + FStart)) max(taxis(FShow*FAvg + FStart))]);
xlim([min(taxis(FShow*FAvg + FStart)) max(taxis(FShow*FAvg + FStart))]);
ylabel({'Centroid along','x (px)'})
xlabel(('Time (s)'))
set(gca,'Position',get(gca,'Position') - [-0.02 0 0.05 0]);
ylim([40 150]);

subplot(3,2,2);
scatter(180/pi*MeanPhase(FShow), MeanDisp(FShow),20,taxis(FShow*FAvg + FStart),'filled');
set(gca,'XColor','k','YColor','k','FontName','Segoe UI','FontSize',15,'LineWidth',1.5);
box on;
colormap(gca,TimeCols);
clim([min(taxis(FShow*FAvg + FStart)) max(taxis(FShow*FAvg + FStart))]);
xlabel('Mean direction (^o)');
ylabel('Mean displacement (px)');
ylim([-0.1 2.5]);
xlim([-180 180])

subplot(3,2,4);
scatter(taxis(FShow*FAvg + FStart),MeanDisp(FShow), 20,taxis(FShow*FAvg + FStart),'filled');
set(gca,'XColor','k','YColor','k','FontName','Segoe UI','FontSize',15,'LineWidth',1.5);
box on;
colormap(gca,TimeCols);
clim([min(taxis(FShow*FAvg + FStart)) max(taxis(FShow*FAvg + FStart))]);
ylabel('Mean displacement (px)');
ylim([-0.1 2.5]);

subplot(3,2,6);
scatter(taxis(FShow*FAvg + FStart),180/pi*MeanPhase(FShow), 20,taxis(FShow*FAvg + FStart),'filled');
set(gca,'XColor','k','YColor','k','FontName','Segoe UI','FontSize',15,'LineWidth',1.5);
box on;
colormap(gca,TimeCols);
clim([min(taxis(FShow*FAvg + FStart)) max(taxis(FShow*FAvg + FStart))]);
ylabel('Mean direction (^o)');
xlabel(('Time (s)'))
ylim([-180 180])



DispAbs = squeeze( sqrt( DispGColl(:,:,1,:).^2 +  DispGColl(:,:,2,:).^2 ));
DispAbs(:,1:min(xShow)) = NaN;
DispAbs(1:min(yShow),:) = NaN;
DispAbs(:,max(xShow):numy) = NaN;
DispAbs(max(yShow):numx,:) = NaN;
DispAbs(DispAbs > 10) = NaN;

DispPhase = squeeze( atan2(DispGColl(:,:,1,:),DispGColl(:,:,2,:)) );
DispPhase(:,1:min(xShow)) = NaN;
DispPhase(1:min(yShow),:) = NaN;
DispPhase(:,max(xShow):numy) = NaN;
DispPhase(max(yShow):numx,:) = NaN;

MeanDisp = sum( reshape(DispAbs,numx*numy,[]),1,'omitnan')./(numx*numy);
MeanPhase = median( reshape(DispPhase,numx*numy,[]),1,'omitnan');
CentX =  sum(squeeze(sum(DispAbs,1,'omitnan')).*repmat( (1:numx)',[1 length(MeanDisp)]),'omitnan')./ ...
    sum(squeeze(sum(DispAbs,1,'omitnan')),'omitnan');
CentY =  sum(squeeze(sum(DispAbs,2,'omitnan')).*repmat( (1:numy)',[1 length(MeanDisp)]),'omitnan')./ ...
    sum(squeeze(sum(DispAbs,2,'omitnan')),'omitnan');
% [~,CentX] =max( sum(DispAbs,2,'omitnan'));
% [~,CentY] = max( sum(DispAbs,1,'omitnan'));

CentX(MeanDisp < 0.05) = NaN;
CentY(MeanDisp < 0.05) = NaN;

CentX = movmean(squeeze(CentX),4);
CentY = movmean(squeeze(CentY),4);

figure(303);
set(gcf,'Color',[1 1 1]);
clf;

subplot(2,2,1); hold all;
imagesc(PhaseColl(:,:,1),'AlphaData',0.5)
for idx = FShow
    scatter(squeeze(CentX(idx)),squeeze(CentY(idx)),...
    20,taxis(idx*FAvg + FStart),'filled','MarkerFaceAlpha',0.8,'MarkerEdgeColor','none','MarkerFaceColor',TimeCols(round((idx - min(FShow))/5) + 1,:));

    plot(CentX(idx)+[0 -4*(MeanDisp(idx)+2)*sin(MeanPhase(idx))],CentY(idx)+[0 -4*(MeanDisp(idx)+2)*cos(MeanPhase(idx))],...
        'Color',[TimeCols(round((idx - min(FShow))/5) + 1,:) 0.8]);
end
set(gca,'XColor','k','YColor','k','FontName','Segoe UI','FontSize',15,'LineWidth',1.5,'YDir','reverse','XTick',[],'YTick',[],'Color','w');
colormap(gca,SPoOFMap_colorcet('C5'));
% colormap(gca,TimeCols);
% c = colorbar('Location','EastOutside','LineWidth',1.5,'Color','k');
% c.Label.String = 'Time (s)';
box on;
axis image;
ylim([min(yShow) max(yShow)]);
xlim([min(xShow) max(xShow)]);
clim([-pi pi])
% clim([min(taxis(FShow*FAvg + FStart)) max(taxis(FShow*FAvg + FStart))]);
% xlabel(('x (px)'))
% ylabel(('y (px)'))
title({'Displacement vectors at their ','centroids along time'})

subplot(4,2,5); hold all;
scatter(taxis(FShow*FAvg + FStart),squeeze(CentY(FShow)),...
    20,taxis(FShow*FAvg + FStart),'filled');
set(gca,'XColor','k','YColor','k','FontName','Segoe UI','FontSize',15,'LineWidth',1.5,'YDir','normal');
box on;
colormap(gca,TimeCols);
clim([min(taxis(FShow*FAvg + FStart)) max(taxis(FShow*FAvg + FStart))]);
xlim([min(taxis(FShow*FAvg + FStart)) max(taxis(FShow*FAvg + FStart))]);
ylabel({'Centroid along','y (px)'})
% xlabel(('Time (s)'))
set(gca,'Position',get(gca,'Position') - [-0.02 0 0.05 0]);
ylim([40 200]);

subplot(4,2,7); hold all;
scatter(taxis(FShow*FAvg + FStart),squeeze(CentX(FShow)),...
    20,taxis(FShow*FAvg + FStart),'filled');
set(gca,'XColor','k','YColor','k','FontName','Segoe UI','FontSize',15,'LineWidth',1.5,'YDir','normal');
box on;
colormap(gca,TimeCols);
clim([min(taxis(FShow*FAvg + FStart)) max(taxis(FShow*FAvg + FStart))]);
xlim([min(taxis(FShow*FAvg + FStart)) max(taxis(FShow*FAvg + FStart))]);
ylabel({'Centroid along','x (px)'})
xlabel(('Time (s)'))
set(gca,'Position',get(gca,'Position') - [-0.02 0 0.05 0]);
ylim([40 150]);

subplot(3,2,2);
scatter(180/pi*MeanPhase(FShow), MeanDisp(FShow),20,taxis(FShow*FAvg + FStart),'filled');
set(gca,'XColor','k','YColor','k','FontName','Segoe UI','FontSize',15,'LineWidth',1.5);
box on;
colormap(gca,TimeCols);
clim([min(taxis(FShow*FAvg + FStart)) max(taxis(FShow*FAvg + FStart))]);
xlabel('Mean direction (^o)');
ylabel('Mean displacement (px)');
ylim([-0.1 2.5]);
xlim([-180 180])

subplot(3,2,4);
scatter(taxis(FShow*FAvg + FStart),MeanDisp(FShow), 20,taxis(FShow*FAvg + FStart),'filled');
set(gca,'XColor','k','YColor','k','FontName','Segoe UI','FontSize',15,'LineWidth',1.5);
box on;
colormap(gca,TimeCols);
clim([min(taxis(FShow*FAvg + FStart)) max(taxis(FShow*FAvg + FStart))]);
ylabel('Mean displacement (px)');
ylim([-0.1 2.5]);

subplot(3,2,6);
scatter(taxis(FShow*FAvg + FStart),180/pi*MeanPhase(FShow), 20,taxis(FShow*FAvg + FStart),'filled');
set(gca,'XColor','k','YColor','k','FontName','Segoe UI','FontSize',15,'LineWidth',1.5);
box on;
colormap(gca,TimeCols);
clim([min(taxis(FShow*FAvg + FStart)) max(taxis(FShow*FAvg + FStart))]);
ylabel('Mean direction (^o)');
xlabel(('Time (s)'))
ylim([-180 180])
%%
figure(302);
print(gcf,'Beat2_PhaseDisp.jpg','-djpeg','-r600')
savefig(gcf,'Beat2_PhaseDisp.fig')

figure(303);
print(gcf,'Beat2_RetDisp.jpg','-djpeg','-r600')
savefig(gcf,'Beat2_RetDisp.fig')


%%

figure(304);
set(gcf,'Color',[1 1 1]-1,'InvertHardCopy','off')
clf;
ArrowGap = 7;

for TFShow = 150:50:400
clf;
subplot(2,1,1); hold all;
%     imagesc(sqrt(DispF(:,:,1).^2 + DispF(:,:,2).^2));
imagesc(180/pi*atan2(DispFColl(:,:,2,TFShow),DispFColl(:,:,1,TFShow)),...
    'AlphaData',~isnan(mean(DispFColl(:,:,:,TFShow),3)));
quiver(Xu(3:ArrowGap:end,3:ArrowGap:end),Yu(3:ArrowGap:end,3:ArrowGap:end),...
    2*DispFColl(3:ArrowGap:end,3:ArrowGap:end,1,TFShow),...
    2* DispFColl(3:ArrowGap:end,3:ArrowGap:end,2,TFShow),0,'w','LineWidth',0.25);
text(10,10,[num2str(taxis((TFShow)*FAvg + 1 + FStart),'%1.3f') ' s'],'FontName','Segoe UI','FontSize',20,'Color','y','FontWeight','bold');
axis image;
set(gca,'XColor','w','YColor','w','XTick',[],'YTick',[],'FontName','Segoe UI','FontSize',15,'Color','k','YDir','reverse');
colormap(gca,SPoOFMap_colorcet('C1'));
ylim([min(yShow) max(yShow)]);
xlim([min(xShow) max(xShow)]);
clim([-pi pi]*180/pi);
box on;
c = colorbar('Location','EastOutside','LineWidth',1.5,'Color','w');
c.Label.String = 'Displacement map for phase (^o)';


subplot(2,1,2); hold all;
imagesc(180/pi*atan2(DispGColl(:,:,2,TFShow),DispGColl(:,:,1,TFShow)),...
    'AlphaData',~isnan(mean(DispGColl(:,:,:,TFShow),3)));
quiver(Xu(3:ArrowGap:end,3:ArrowGap:end),Yu(3:ArrowGap:end,3:ArrowGap:end),...
    2*DispGColl(3:ArrowGap:end,3:ArrowGap:end,1,TFShow), ...
    2*DispGColl(3:ArrowGap:end,3:ArrowGap:end,2,TFShow),0,'w','LineWidth',0.25)
axis image;
set(gca,'XColor','w','YColor','w','XTick',[],'YTick',[],'FontName','Segoe UI','FontSize',15,'Color','k','YDir','reverse');
ylim([min(yShow) max(yShow)]);
xlim([min(xShow) max(xShow)]);
c = colorbar('Location','EastOutside','LineWidth',1.5,'Color','w');
c.Label.String = 'Displacement map for retardation angle (^o)';
clim([-pi pi]*180/pi);
colormap(gca,SPoOFMap_colorcet('C1'));

box on;
%%
Figgg = getframe(gcf);
Figgg.cdata = Figgg.cdata(:,96:425,:);
imwrite(Figgg.cdata,['DispField_' num2str(taxis((TFShow)*FAvg + 1 + FStart),'%1.3f') ' s.jpg'],'Quality',100);
savefig(gcf,['DispField_' num2str(taxis((TFShow)*FAvg + 1 + FStart),'%1.3f') ' s.fig'])
%%
end

Figgg = getframe(gcf);
imwrite(Figgg.cdata,['DispField_' num2str(taxis((TFShow)*FAvg + 1 + FStart),'%1.3f') ' s.jpg'],'Quality',100);
savefig(gcf,['DispField_' num2str(taxis((TFShow)*FAvg + 1 + FStart),'%1.3f') ' s.fig'])