FID1 = fopen([pwd '\' fbase '\' OutName1 '.bin'],'rb');
FID2 = fopen([pwd '\' fbase '\' OutName2 '.bin'],'rb');

fseek(FID1,numx*numy*2*29500*4,'bof');
fseek(FID2,numx*numy*2*29500*4,'bof');

NumImages = 6;
BigImages = zeros([numx numy NumImages 3]);

for idx = 1:NumImages

    fseek(FID1,numx*numy*2*250*4,'cof');
    Out_P1 = fread(FID1,numx*numy*2,'float');
    Out_P1 = reshape( Out_P1(1:2:end-1) + 1i.*Out_P1(2:2:end),[numx numy]).*exp(-1i*PhaseMask1);    
    
    fseek(FID2,numx*numy*2*250*4,'cof');
    Out_P2 = fread(FID2,numx*numy*2,'float');
    Out_P2 = reshape( Out_P2(1:2:end-1) + 1i.*Out_P2(2:2:end),[numx numy]).*exp(-1i*PhaseMask2);

    Ret = atan2( abs( 1i.*Out_P2.*exp(1i.*pi/2) - Out_P1), abs( 1i.*Out_P2.*exp(1i.*pi/2) + Out_P1));

    BigImages(:,:,idx,1) = Out_P1;
    BigImages(:,:,idx,2) = Out_P2;
    BigImages(:,:,idx,3) = Ret;

end

fclose(FID1);
fclose(FID2);

%%
figure(1); clf;
subplot(5,1,[1 2]); hold all;
imagesc([reshape( abs(BigImages(:,:,:,1)),[numx numx*NumImages]);reshape( abs(BigImages(:,:,:,2)),[numx numx*NumImages])].^0.7);
colormap(gca,OCMIntMap);
for idx = 1:NumImages
    plot(numx*idx*[1 1],[1 2*numx],'Color','w','LineWidth',4);
    text(numx*idx-(numx-30),2*(numx-7),num2str(29.50 + idx*0.25,'t = %1.2f s'),'FontName','Segoe UI','FontSize',18,'Color','w','FontWeight','bold')
end
plot([numx*NumImages 1],[numx numx],'Color','w','LineWidth',4)
set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI','FontSize',25,'YDir','reverse');
clim([100 350]);
c = colorbar('Location','EastOutside','LineWidth',1.5);
axis image;
c.Label.String = 'Intensity (a.u.)';

subplot(5,1,[3 4]); hold all;
imagesc([reshape( angle(BigImages(:,:,:,1)),[numx numx*NumImages]);reshape( angle(BigImages(:,:,:,2)),[numx numx*NumImages])],...
    'AlphaData',([reshape( abs(BigImages(:,:,:,1)),[numx numx*NumImages]);reshape( abs(BigImages(:,:,:,2)),[numx numx*NumImages])].^0.7 - 3)/20);
for idx = 1:NumImages
    plot(numx*idx*[1 1],[1 2*numx],'Color','w','LineWidth',4)
end
plot([numx*NumImages 1],[numx numx],'Color','w','LineWidth',4)
colormap(gca,OCMPhaseMap)
set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI','Color',[0 0 0],'FontSize',25);
c = colorbar('Location','EastOutside','LineWidth',1.5);
c.Label.String = 'Phase (rad)';
axis image;


subplot(5,1,5); hold all;
imagesc([reshape( BigImages(:,:,:,3),[numx numx*NumImages])]);
for idx = 1:NumImages
    plot(numx*idx*[1 1],[1 numx],'Color','w','LineWidth',4)
end
plot([numx*NumImages 1],[numx numx],'Color','w','LineWidth',4)
colormap(gca,SPoOFMap_colorcet('R2'))
set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI','Color',[0 0 0],'FontSize',25);
c = colorbar('Location','EastOutside','LineWidth',1.5);
c.Label.String = 'Retardation (rad)';
axis image;
% clim([1 100]);


%%
ROI1 = 85;
ROI2 = 90;

zoomsize = 40;

figure(1); clf;
subplot(5,1,[1 2]); hold all;
imagesc([reshape( abs(BigImages(ROI1+(1:zoomsize),ROI2+(1:zoomsize),:,1)),[zoomsize zoomsize*NumImages]);...
    reshape( abs(BigImages(ROI1+(1:zoomsize),ROI2+(1:zoomsize),:,2)),[zoomsize zoomsize*NumImages]);].^0.7);
colormap(gca,OCMIntMap);
for idx = 1:NumImages
    plot(zoomsize*idx*[1 1],[1 2*zoomsize],'Color','w','LineWidth',4);
%     text(numx*idx-(numx-30),2*(numx-7),num2str(29.50 + idx*0.25,'t = %1.2f s'),'FontName','Segoe UI','FontSize',18,'Color','w','FontWeight','bold')
end
plot([zoomsize*NumImages 1],[zoomsize zoomsize],'Color','w','LineWidth',4)
set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI','FontSize',25,'YDir','reverse');
clim([100 350]);
c = colorbar('Location','EastOutside','LineWidth',1.5);
axis image;
c.Label.String = 'Intensity (a.u.)';

subplot(5,1,[3 4]); hold all;
imagesc([reshape( angle(BigImages(ROI1+(1:zoomsize),ROI2+(1:zoomsize),:,1)),[zoomsize zoomsize*NumImages]);...
    reshape( angle(BigImages(ROI1+(1:zoomsize),ROI2+(1:zoomsize),:,2)),[zoomsize zoomsize*NumImages]);],...
    'AlphaData',([reshape( abs(BigImages(ROI1+(1:zoomsize),ROI2+(1:zoomsize),:,1)),[zoomsize zoomsize*NumImages]);...
    reshape( abs(BigImages(ROI1+(1:zoomsize),ROI2+(1:zoomsize),:,2)),[zoomsize zoomsize*NumImages]);].^0.7 - 3)/20);
for idx = 1:NumImages
    plot(zoomsize*idx*[1 1],[1 2*zoomsize],'Color','w','LineWidth',4);
%     text(numx*idx-(numx-30),2*(numx-7),num2str(29.50 + idx*0.25,'t = %1.2f s'),'FontName','Segoe UI','FontSize',18,'Color','w','FontWeight','bold')
end
plot([zoomsize*NumImages 1],[zoomsize zoomsize],'Color','w','LineWidth',4)
colormap(gca,OCMPhaseMap)
set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI','Color',[0 0 0],'FontSize',25);
c = colorbar('Location','EastOutside','LineWidth',1.5);
c.Label.String = 'Phase (rad)';
axis image;


subplot(5,1,5); hold all;
imagesc(reshape( (BigImages(ROI1+(1:zoomsize),ROI2+(1:zoomsize),:,3)),[zoomsize zoomsize*NumImages]));
for idx = 1:NumImages
    plot(zoomsize*idx*[1 1],[1 zoomsize],'Color','w','LineWidth',4);
%     text(numx*idx-(numx-30),2*(numx-7),num2str(29.50 + idx*0.25,'t = %1.2f s'),'FontName','Segoe UI','FontSize',18,'Color','w','FontWeight','bold')
end
plot([zoomsize*NumImages 1],[zoomsize zoomsize],'Color','w','LineWidth',4)
colormap(gca,SPoOFMap_colorcet('R2'))
set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI','Color',[0 0 0],'FontSize',25);
c = colorbar('Location','EastOutside','LineWidth',1.5);
c.Label.String = 'Retardation (rad)';
axis image;