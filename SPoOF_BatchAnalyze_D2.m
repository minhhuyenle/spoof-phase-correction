%%
clear all;
OCMIntMap = OCMMap_viridis(256);
OCMPhaseMap = circshift(SPoOFMap_colorcet('C3'),128);
OCMDivMap = SPoOFMap_colorcet('D5');
%%

FDir = dir('*512x512*');

FcxOff = [0 -1 -1 0  0 0 0 0];
FcyOff = [4  2  2  2 0 0 0 0];
%%

for fileidx = 1:length(FDir)
%%
    tStarttic = tic;
    fbase = FDir(fileidx).name;
    disp(fbase);
    
 %%   
    bbase = 'INVALID';
    OutName1 = 'Processed_P1';
    OutName2 = 'Processed_P2';
    numX = 512;
    numY = 512;
    
    numF = floor(dir([pwd '\' fbase '\' fbase '.mraw']).bytes/(numX*numY*1.5));
    
    % numF = 43000;
    
    numx = 200;
    numy = 200;
    
    FCx1 = 32 - FcxOff(fileidx);
    FCy1 = 53 - FcyOff(fileidx);
    
    Rad = 100;
    
    FCx2 = 27 + FcxOff(fileidx);
    FCy2 = 262 + FcyOff(fileidx);
    
    [X, Y] = meshgrid( linspace(-1,1,numx),linspace(-1,1,numy));
    RHO = sqrt(X.^2  + Y.^2);
    Win = (RHO.^2 <=1);
    
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
    if fileidx > 1
        system(['D:\Rishee\SPoOFOCM_Proc\SPoOFOCM_PostProc\x64\Debug\SPoOFOCM_PostProc.exe ' pwd '\Config.txt']);
    end
    
    %
    % bbase = 'Back_200.00usExp_4000fps_24';
    OutName1 = 'Processed_P1';
    OutName2 = 'Processed_P2';
    
%     numF = 43674;
    
%     numx = 200;
%     numy = 200;
%%    
    FID1 = fopen([pwd '\' fbase '\' OutName1 '.bin'],'rb');
    fseek(FID1,numx*numy*2*1200*4,'bof');
    Out_P1 = fread(FID1,numx*numy*2,'float');
    Out_P1 = reshape( Out_P1(1:2:end-1) + 1i.*Out_P1(2:2:end),[numx numy]);
    fclose(FID1);
    
    FID2 = fopen([pwd '\' fbase '\' OutName2 '.bin'],'rb');
    fseek(FID2,numx*numy*2*1200*4,'bof');
    Out_P2 = fread(FID2,numx*numy*2,'float');
    Out_P2 = reshape( Out_P2(1:2:end-1) + 1i.*Out_P2(2:2:end),[numx numy]);
    fclose(FID2);
   
    
    OCMPhaseMap = circshift( SPoOFMap_colorcet('C3'), 0);
    
    RaiMap = [linspace(0,0.4,64)' linspace(0,0,64)' linspace(0,0.5,64)';...
        linspace(0.4,1,64)' linspace(0,0,64)' linspace(0.5,0.5,64)';...
        linspace(1,1,64)' linspace(0,1,64)' linspace(0.5,0.5,64)';...
        linspace(1,1,64)' linspace(1,1,64)' linspace(0.5,1,64)';...
        ];
    
    Sigma = 0.04;
    [X, Y] = meshgrid( linspace(-1,1,numx),linspace(-1,1,numy));
    RHO = sqrt(X.^2  + Y.^2);
    G2D = exp( - ( RHO.*RHO)/(2.*Sigma.^2));
    G2D = G2D / max(G2D(:));
    G2D = 1 - G2D;
    Win = (RHO.^2 <=2);
    
    y1 = 0;
    y2 = 80;
    x1 = -15;
    x2 = 75;
    xyc = -1.5;
    ycen = 0;
    xcen = 0;
    off = 0;
    [x,y] = meshgrid( linspace(-1,1,numx),linspace(-1,1,numy));
    PhaseMask1 = y1.*y + x1.*x + y2.*(y-ycen).^2 + x2.*(x-xcen).^2 + xyc.*x.*y + off;
    
    y1 = 0;
    y2 = -80;
    x1 = 14;
    x2 = -78;
    xyc = -13;
    ycen = 0;
    xcen = 0;
    off = 3;
    [x,y] = meshgrid( linspace(-1,1,numx),linspace(-1,1,numy));
    PhaseMask2 = y1.*y + x1.*x + y2.*(y-ycen).^2 + x2.*(x-xcen).^2 + xyc.*x.*y + off;
    
    figure(1);
    clf;
    set(gcf,'Color',[1 1 1]);
    subplot(321);
    hold all;
    imagesc( abs( Out_P1'/3).^0.8 .* Win,'AlphaData',Win);
    colormap(gca,OCMIntMap);
    axis image;
    % colorbar;
    clim([0 1]*1500)
    set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI');
    title('|E_{P1}|','Color',[1 1 1]*0);
    colorbar('Location','EastOutside','LineWidth',1.5);
    
    
    subplot(322);
    hold all;
    cla;
    imagesc( abs( Out_P2'/2).^0.8 .* Win,'AlphaData',Win);
    colormap(gca,OCMIntMap);
    clim([0 1]*1500)
    set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI');
    title('|E_{P2}|','Color',[1 1 1]*0);
    axis image;
    colorbar('Location','EastOutside','LineWidth',1.5);
    
    
    subplot(323);
    imagesc( Win.*-angle(exp(-1i*PhaseMask1).*Out_P1)','AlphaData', Win);
    colormap(gca,OCMPhaseMap);
    axis image;
    set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI');
    title('\angleE_{P1}','Color',[1 1 1]*0);
    clim([-pi pi]);
    c = colorbar('Location','EastOutside','LineWidth',1.5,'Ticks',[-pi -pi/2 0 pi/2 pi],'TickLabels',{'-π','-π/2','0','π/2','π'});
    
    subplot(324);
    imagesc( Win.* angle(exp(-1i*PhaseMask2).*Out_P2)','AlphaData', Win);
    colormap(gca,OCMPhaseMap);
    axis image;
    set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI');
    title('\angleE_{P2}','Color',[1 1 1]*0);
    clim([-pi pi]);
    c = colorbar('Location','EastOutside','LineWidth',1.5,'Ticks',[-pi -pi/2 0 pi/2 pi],'TickLabels',{'-π','-π/2','0','π/2','π'});
    
    
    subplot(325);
    imagesc( Win.*angle(exp(-1i*PhaseMask1).*Out_P1.*exp(-1i*PhaseMask2).*Out_P2)','AlphaData', Win);
    colormap(gca,OCMDivMap);
    axis image;
    set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI');
    title('\angleE_{P1} - \angleE_{P2}','Color',[1 1 1]*0);
    c = colorbar('Location','EastOutside','LineWidth',1.5,'Ticks',[-pi -pi/2 0 pi/2 pi],'TickLabels',{'-π','-π/2','0','π/2','π'});
    clim([-pi pi]*0.5)
    
    
    subplot(326);
    imagesc( Win.*(angle(exp(-1i*PhaseMask1).*Out_P1.*exp(1i*PhaseMask2).*conj(Out_P2))'),'AlphaData', Win);
    colormap(gca,OCMPhaseMap);
    axis image;
    set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI');
    title('\angleE_{P1} + \angleE_{P2}','Color',[1 1 1]*0);
    clim([-pi pi]);
    c = colorbar('Location','EastOutside','LineWidth',1.5,'Ticks',[-pi -pi/2 0 pi/2 pi],'TickLabels',{'-π','-π/2','0','π/2','π'});
    %%
    savefig(gcf,[pwd '\' fbase '\JustBeautifulImages_9NOV2023.fig']);
    
    %%
    DFWin = 4;
    Out_P1_DF = fftshift(fft2(Out_P1)).*G2D;
    Out_P1_DF(end/2-DFWin+1:end/2+DFWin,end/2-DFWin+1:end/2+DFWin) = 0;
    Out_P1_DF = (ifft2(ifftshift(Out_P1_DF)));
    
    Out_P2_DF = fftshift(fft2(Out_P2)).* G2D;
    Out_P2_DF(end/2-DFWin+1:end/2+DFWin,end/2-DFWin+1:end/2+DFWin) = 0;
    Out_P2_DF = (ifft2(ifftshift(Out_P2_DF)));
    
    AbsWin = 4;
    
    ElecLoc = [0 0];
    Orig = [0 0];
    
    Gap = 5;
    
    [LocListX, LocListY] = meshgrid( ((AbsWin/2+1):Gap:(numx-(AbsWin/2+1)))+Orig(1),...
        ((AbsWin/2+1):Gap:(numx-(AbsWin/2+1)))+Orig(2));
    LocList = [LocListX(:) LocListY(:)];
    
    DistToElec = sqrt( (LocList(:,1) - ElecLoc(1)).^2 + (LocList(:,2) - ElecLoc(2)).^2);
    [~,I] = sort(DistToElec);
    
    XL = LocList(I,1)-AbsWin/2;
    YL = LocList(I,2)-AbsWin/2;
    
    OCMPhaseMap2 = circshift(SPoOFMap_colorcet('C3'),128);
    ROIColors = SPoOFMap_colorcet('I1','N',length(XL)).^2;
    if length(XL) > 256
       ColorETemp = ROIColors;
       ROIColors = repmat(ROIColors,[ceil(length(XL)/256) 1]);
       for ccccidx = 1:ceil(length(XL)/256)
            ROIColors(ccccidx:ceil(length(XL)/256):end,:) = ColorETemp;
       end
    end
        
    figure(2);
    set(gcf,'Color',[1 1 1]);
    clf;
    subplot(221);
    hold all;
    imagesc( abs( Out_P1_DF'/2).^0.8.*Win,'AlphaData', Win*0.4);
    colormap(gca,OCMIntMap);
    axis image;
    clim([0.1 1]*1000)
    set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI');
    title('|E_{P1}|','Color',[1 1 1]*0);
    for lidx = 1:length(XL)
        plot([YL(lidx)+AbsWin YL(lidx)+AbsWin],[XL(lidx) XL(lidx)+AbsWin],'Color',ROIColors(lidx,:),'LineWidth',1.5);
        plot([YL(lidx) YL(lidx)],[XL(lidx) XL(lidx)+AbsWin],'Color',ROIColors(lidx,:),'LineWidth',1.5);
        plot([YL(lidx) YL(lidx)+AbsWin],[XL(lidx) XL(lidx)],'Color',ROIColors(lidx,:),'LineWidth',1.5);
        plot([YL(lidx) YL(lidx)+AbsWin],[XL(lidx)+AbsWin XL(lidx)+AbsWin],'Color',ROIColors(lidx,:),'LineWidth',1.5);
    %     text(YL(lidx)-8, XL(lidx)-8, num2str(lidx),'Color',ROIColors(lidx,:),'FontSize',12,'FontName','Segoe UI');
    end
    colorbar('Location','EastOutside','LineWidth',1.5);
    
    subplot(222);
    hold all;
    cla;
    imagesc( abs( Out_P2_DF'/2).^0.8.*Win,'AlphaData', Win);
    colormap(gca,OCMIntMap);
    clim([0.1 1]*1000)
    set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI');
    title('|E_{P2}|','Color',[1 1 -2]*0);
    axis image;
    % for lidx = 1:length(XL)
    %     plot([YL(lidx)+AbsWin YL(lidx)+AbsWin],[XL(lidx) XL(lidx)+AbsWin],'Color',ROIColors(lidx,:),'LineWidth',1.5);
    %     plot([YL(lidx) YL(lidx)],[XL(lidx) XL(lidx)+AbsWin],'Color',ROIColors(lidx,:),'LineWidth',1.5);
    %     plot([YL(lidx) YL(lidx)+AbsWin],[XL(lidx) XL(lidx)],'Color',ROIColors(lidx,:),'LineWidth',1.5);
    %     plot([YL(lidx) YL(lidx)+AbsWin],[XL(lidx)+AbsWin XL(lidx)+AbsWin],'Color',ROIColors(lidx,:),'LineWidth',1.5);
    %     text(YL(lidx)-5, XL(lidx)-5, num2str(lidx),'Color',ROIColors(lidx,:),'FontSize',12,'FontName','Segoe UI');
    % end
    colorbar('Location','EastOutside','LineWidth',1.5);
    
    OCMDivMap = SPoOFMap_colorcet('D5');
    
    subplot(223);
    hold all; cla;
    imagesc( -angle(exp(-1i*PhaseMask1).*Out_P1)','AlphaData', Win*0.4);
    colormap(gca,OCMPhaseMap);
    axis image;
    set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI','Color',[1 1 1]);
    title('\angleE_{P1}','Color',[1 1 1]*0);
    for lidx = 1:length(XL)
        plot([YL(lidx)+AbsWin YL(lidx)+AbsWin],[XL(lidx) XL(lidx)+AbsWin],'Color',ROIColors(lidx,:),'LineWidth',2);
        plot([YL(lidx) YL(lidx)],[XL(lidx) XL(lidx)+AbsWin],'Color',ROIColors(lidx,:),'LineWidth',2);
        plot([YL(lidx) YL(lidx)+AbsWin],[XL(lidx) XL(lidx)],'Color',ROIColors(lidx,:),'LineWidth',2);
        plot([YL(lidx) YL(lidx)+AbsWin],[XL(lidx)+AbsWin XL(lidx)+AbsWin],'Color',ROIColors(lidx,:),'LineWidth',2);
    %     if rem(lidx,10) == 1
    %         text(YL(lidx)-8, XL(lidx)+14, num2str(lidx),'Color',ROIColors(lidx,:),'FontSize',15,'FontName','Segoe UI');
    %     end
    end
    c = colorbar('Location','EastOutside','LineWidth',1.5,'Ticks',[-pi -pi/2 0 pi/2 pi],'TickLabels',{'-π','-π/2','0','π/2','π'});
    
    subplot(224);
    imagesc( angle(exp(-1i*PhaseMask2).*Out_P2)','AlphaData', Win);
    colormap(gca,OCMPhaseMap);
    axis image;
    set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI');
    title('\angleE_{P2}','Color',[1 1 1]*0);
    c = colorbar('Location','EastOutside','LineWidth',1.5,'Ticks',[-pi -pi/2 0 pi/2 pi],'TickLabels',{'-π','-π/2','0','π/2','π'});
    
    savefig(gcf,[pwd '\' fbase '\' fbase '_ROIPlots_9NOV2023.fig']);
    
    %%
    
    FID1 = fopen([pwd '\' fbase '\' OutName1 '.bin'],'rb');
    FID2 = fopen([pwd '\' fbase '\' OutName2 '.bin'],'rb');
    
    numFrames = numF;
    T1 = rand([length(XL) numFrames]) + 1i*rand([length(XL) numFrames]);
    T2 = rand([length(XL) numFrames]) + 1i*rand([length(XL) numFrames]);
    wBar = waitbar(0,'Starting...');
    tic;
    for frameIdx = 1:numFrames
        Out_P1 = fread(FID1,numx*numy*2,'float');
        Out_P1 = reshape( Out_P1(1:2:end-1) + 1i.*Out_P1(2:2:end),[numx numy]);
    
        Out_P2 = fread(FID2,numx*numy*2,'float');
        Out_P2 = reshape( Out_P2(1:2:end-1) + 1i.*Out_P2(2:2:end),[numx numy]);
    
        Out_P1 = exp(-1i*PhaseMask1).*Out_P1;
        Out_P2 = exp(-1i*PhaseMask2).*Out_P2;
        for lidx = 1:length(XL)
            ResP1 = Out_P1(YL(lidx):YL(lidx)+AbsWin,XL(lidx):XL(lidx)+AbsWin);
            ResP2 = Out_P2(YL(lidx):YL(lidx)+AbsWin,XL(lidx):XL(lidx)+AbsWin);
    
            T1(lidx,frameIdx) = mean(ResP1(:));
            T2(lidx,frameIdx) = conj(mean(ResP2(:))).*exp(1i.*pi/2);
        end
        if rem(frameIdx,100) == 0
            waitbar(frameIdx/numFrames,wBar,['Loaded ' num2str(frameIdx) ' frames in ' num2str(toc,'%1.0f') ' s...']);
        end
    end
    close(wBar);
    fclose(FID2);
    fclose(FID1);
    % Butterworth filtered P1 and P2 phase Plots
    
    %%
    figure(7);
    clf; hold all;
    set(gcf,'Color',[1 1 1]);
    set(gca,'LineWidth',1.5,'FontSize',15,'FontName','Segoe UI');
    %%
    
    
    FileDetails.fbase = fbase;
    FileDetails.fDEEET = 'Glut'; 
    FileDetails.FileExt = '9NOV2023';
    N = length(XL);
    
    

    Stim.ONT = 0;
    fs = 1000;
    
    taxis = (0:(numFrames-1))/fs;

    Filt.fop = 4;
    Filt.PlotOption = 10;
    
    Filt.AbsWin = AbsWin;
    Filt.PhaseOff = 0;
    Filt.NotchOn = 1;
    
    if Filt.fop == 1
        fc = [50 70];
        [b,a] = butter(3,fc/(fs/2),'stop');
        FiltOpt = '60HzF';
    elseif Filt.fop == 2
        FiltOpt = '1000HzLPF';
        [b,a] = butter(6,1000/(fs/2),'low');
    elseif Filt.fop == 3
        FiltOpt = '70HzHPF';
        [b,a] = butter(6,70/(fs/2),'high');  
    elseif Filt.fop == 4
        FiltOpt = 'Gauss';
        windowSize = 10; 
        b = gausswin(windowSize,2);%(1/windowSize)*ones(1,windowSize);
        b = b/sum(b);
        a = 1;
    elseif Filt.fop == 5
        FiltOpt = 'Rect';
        windowSize = 10; 
        b = (1/windowSize)*ones(1,windowSize);
        a = 1;
    end
    
    if Filt.NotchOn == 1
         [bNotch,aNotch] = butter(4,[110 130]/(fs/2),'stop');
         [bNotch2,aNotch2] = butter(2,[7 10]/(fs/2),'stop');
    end
    tic;
    
    delta_s = atan2( abs( 1i.*T2 - T1), abs( 1i.*T2 + T1));
    disp([num2str(toc,'%1.2f') ' s...'])
    
    T3_N = filtfilt( b,a, delta_s')*180/pi;
    disp([num2str(toc,'%1.2f') ' s...'])
    if Filt.NotchOn == 1
        T3_N = filtfilt(bNotch, aNotch, T3_N);
        T3_N = filtfilt(bNotch2, aNotch2, T3_N);
    end
    disp([num2str(toc,'%1.2f') ' s...'])
    
    %             TFilt = (TFilt - filter( bMean,aMean, TFilt));
    T3_N = (T3_N - mean(T3_N(2000:40000,:),1));
    disp([num2str(toc,'%1.2f') ' s...' num2str(sum(isnan(T3_N(:)))./length(T3_N(:)))])
    
    
    T3ChartNorm = T3_N;
    T3ChartNorm(:,1:end) = T3_N(:,1:end) - 0*median(T3_N(:,2:end),2,'omitnan');
    T3CharMovMean = movmean(T3ChartNorm,25,1);
    %%
    figure(7);
    
    Filt.Gap = 0.5;
    ColorPlotMap = [ROIColors];
    clf;
    set(gcf,'Color',[1 1 1]);
    
    subplot(1,4,1)
    hold all;
    
    for lidx = 1:N
        if(var(T3ChartNorm(:,lidx)) > 40)
            T3ChartNorm(:,lidx) = 0;
        end
    end
    
    for lidx = [5:40:N]
        plot(taxis*1000,T3ChartNorm(:,lidx) + Filt.Gap*lidx,'Color',ColorPlotMap(lidx,:),...
            'LineWidth',1);   
        if sum( isnan( T3ChartNorm(:,lidx))) > 0
            plot(taxis*1000, Filt.Gap*lidx * ones( size( taxis)),'Color',ColorPlotMap(lidx,:),...
            'LineWidth',1);
        end    
        if rem(lidx,200) == 5 && lidx > 1
           text(43500, Filt.Gap*lidx,...
               ['ROI ' num2str(lidx)],'FontSize',13,'FontName','Segoe UI');
        end
        pause(0.1);
    end
    xlim([200 44000]);
    set(gca,'LineWidth',1.5,'FontSize',15,'FontName','Segoe UI','YDir','normal','YTick',[],'YColor',[1 1 1]);
    
    subplot(1,4,2)
    hold all;
    for lidx = [5:40:N]
        plot(taxis*1000,T3CharMovMean(:,lidx) + Filt.Gap*lidx,'Color',ColorPlotMap(lidx,:),...
            'LineWidth',1);   
        if sum( isnan( T3ChartNorm(:,lidx))) > 0
            plot(taxis*1000, Filt.Gap*lidx * ones( size( taxis)),'Color',ColorPlotMap(lidx,:),...
            'LineWidth',1);
        end    
        if rem(lidx,200) == 0 && lidx > 1
           text(6200, Filt.Gap*lidx,...
               ['ROI ' num2str(lidx)],'FontSize',13,'FontName','Segoe UI');
        end
        pause(0.1);
    end
    xlim([200 44000]);
    set(gca,'LineWidth',1.5,'FontSize',15,'FontName','Segoe UI','YDir','normal','YTick',[],'YColor',[1 1 1]);
    
    
    % Filt < do movmean here
    TimeStdWinSize = 25;
    Tstart = 0;
    Tend = floor((numF)/TimeStdWinSize)*TimeStdWinSize;
    
    NTbins = (Tend - Tstart)/TimeStdWinSize;
    
    TStd_norm = zeros([NTbins N]);
    
    for tidx = 1:NTbins
    %     TStd_norm(tidx,:) = mean( T3ChartNorm( Tstart+(tidx-1)*TimeStdWinSize+(1:TimeStdWinSize),1:end-1),1);
          TStd_norm(tidx,:) = median( T3CharMovMean( Tstart+(tidx-1)*TimeStdWinSize+(1:TimeStdWinSize),1:end),1);
    %     TStd_norm(tidx,:) = max(T3ChartNorm( Tstart+(tidx-1)*TimeStdWinSize+(1:TimeStdWinSize),1:end-1),[],1) - ...
    %         min(T3ChartNorm( Tstart+(tidx-1)*TimeStdWinSize+(1:TimeStdWinSize),1:end-1),[],1);
    end
    
    taxis_std = ((Tstart+1):TimeStdWinSize:Tend)/fs;
    %
    subplot(1,4,3)
    Filt.Gap = 0.3;
    ColorPlotMap = [ROIColors];
    cla;
    hold all;
    set(gcf,'Color',[1 1 1]);
    
    for lidx = 1:N
        if(var(TStd_norm(:,lidx)) > 20)
            TStd_norm(:,lidx) = 0;
        end
    end
    
    for lidx = [5:40:N]
        plot(taxis_std*1000, TStd_norm(:,lidx) + Filt.Gap*0.2*lidx,'Color',ColorPlotMap(lidx,:),...
            'LineWidth',1);
        pause(0.1);
    end
    xlim([200 44000]);
    set(gca,'LineWidth',1.5,'FontSize',15,'FontName','Segoe UI','YDir','normal','YTick',[],'YColor',[1 1 1]);
    % savefig(gcf,[pwd '\' fbase '\' fbase '_Traces_9NOV2023.fig']);
    
    %%
    GapWid = YL(2) - YL(1);
    figure(21); clf;
    StdVids = zeros([sqrt(N) sqrt(N) NTbins]);
    for tidx = 1:NTbins
    
        for idx = 1:N
            StdVids((XL(idx) - min(XL))/GapWid + 1,(YL(idx) - min(YL))/GapWid+1,tidx) = TStd_norm(tidx,idx);
        end
    %     imagesc( StdVids(:,:,tidx));
    %     axis image;
    %     clim([0.4 3]);
    %     colormap(SPoOFMap_colorcet('R3'));
    %     set(gca,'LineWidth',1.5,'FontSize',15,'FontName','Segoe UI','YDir','normal','YTick',[],'XTick',[],'YColor',[1 1 1],'XColor',[1 1 1]);
    %     title( taxis_std(tidx));
    %     pause(0.05);
        
    end
    %%
    figure(22);
    subplot(2,2,1);
    imagesc( median(StdVids,3,'omitnan'));
    % clim([2 4]);
    colormap(SPoOFMap_colorcet('R2'));
    axis image;
    set(gca,'LineWidth',1.5,'FontSize',15,'FontName','Segoe UI','YDir','reverse');
    c = colorbar('Location','EastOutside','LineWidth',1.5);
    c.Label.String = 'Mean response';
    clim([0 1])
    
    subplot(2,2,2);
    imagesc( std(StdVids,0,3));
    % clim([2 4]);
    colormap(SPoOFMap_colorcet('R2'));
    axis image;
    set(gca,'LineWidth',1.5,'FontSize',15,'FontName','Segoe UI','YDir','reverse');
    c = colorbar('Location','EastOutside','LineWidth',1.5);
    c.Label.String = 'Std of response';
    clim([0 1])
    
    subplot(2,2,3);
    imagesc( quantile(StdVids,0.95,3));
    % clim([2 4]);
    colormap(SPoOFMap_colorcet('R2'));
    axis image;
    set(gca,'LineWidth',1.5,'FontSize',15,'FontName','Segoe UI','YDir','reverse');
    c = colorbar('Location','EastOutside','LineWidth',1.5);
    c.Label.String = 'Max response';
    clim([0 2.5])
    savefig(gcf,[pwd '\' fbase '\' fbase '_StdevVideos_9NOV2023.fig']);
    
    %%
    
    [coeff, score, latent,~,explained,meanpca] = pca(T3CharMovMean');
    
    xLToShow = 15;
    yLToShow = 15;
    
    PToShow = find(XL ==  (xLToShow-1)*GapWid + min(XL(:)) & YL == (yLToShow-1)*GapWid + min(YL(:)));
    ReconTStd = score(:,1:10)*coeff(:,1:10)' + meanpca;
    % YLSee = find( );
    
    %  = xLToShow * sqrt(N) + yLToShow;
    
    figure(7);
    subplot(1,4,4); cla;
    hold all;
    for lidx = [5:40:N]
        plot(taxis*1000,ReconTStd(lidx,:) + Filt.Gap*lidx,'Color',ColorPlotMap(lidx,:),...
            'LineWidth',1);   
        pause(0.1);
    end
    xlim([200 44000]);
    set(gca,'LineWidth',1.5,'FontSize',15,'FontName','Segoe UI','YDir','normal','YTick',[],'YColor',[1 1 1]);
    
    %%
    
    WinSizePCAStd = 251;
    
    PCAStd = squeeze( std(reshape(ReconTStd(:,:),[N WinSizePCAStd length(taxis)/WinSizePCAStd]),1,2));
    figure(101); clf;
    hold all;
    for lidx = [5:40:N]
        plot(taxis(1:WinSizePCAStd:end)*1000,PCAStd(lidx,:) + Filt.Gap*0.1*lidx,'Color',ColorPlotMap(lidx,:),...
            'LineWidth',1);   
        pause(0.1);
    end
    xlim([200 44000]);
    set(gca,'LineWidth',1.5,'FontSize',15,'FontName','Segoe UI','YDir','normal','YTick',[],'YColor',[1 1 1]);
    
    
    %%
    
    
    PCAStdImg = zeros([sqrt(N) sqrt(N) length(taxis)/WinSizePCAStd]);
    for idx = 1:N
        PCAStdImg((XL(idx) - min(XL))/GapWid + 1,(YL(idx) - min(YL))/GapWid+1,:) = PCAStd(idx,:);
    end
    
    figure(100);
    clf;
    hold all;
    for tidx = 1:length(taxis)/WinSizePCAStd%1:250
        imagesc( PCAStdImg(:,:,tidx) - 0*mean(reshape(PCAStdImg(:,:,tidx),[],1)));
    %     imagesc( mean(PCAStdImg(:,:,120:140),3));
        colormap(gca,SPoOFMap_colorcet('R2'));
        axis image;
        set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI');
        c = colorbar('Location','EastOutside','LineWidth',1.5);
        clim([0 0.3]);
        title(tidx)
        pause(0.01);
    end
    %%
    
    NumClusters = 5;
    
    CheckCond = NumClusters;
    iter = 1;
    tic;
    while CheckCond == NumClusters
    
        kMeansClus = kmeans(ReconTStd(:,:),NumClusters,'MaxIter',100);
        Clustered = zeros([sqrt(N) sqrt(N)]);
        for idx = 1:N
            Clustered((XL(idx) - min(XL))/GapWid + 1,(YL(idx) - min(YL))/GapWid+1) = kMeansClus(idx);
        end
        ClustMap = SPoOFMap_colorcet('R2','N',NumClusters);
        %%
        AbsWin2 = 5;
        figure(21); clf; 
        set(gcf,'Color',[1 1 1],'InvertHardcopy','off')
    
        %%
        subplot(321); cla;
        hold all;
        imagesc( angle(Out_P1'),'AlphaData',Win*1);
    %     for lidx = 1:length(XL)
    %         plot([YL(lidx)+AbsWin2 YL(lidx)+AbsWin2],[XL(lidx) XL(lidx)+AbsWin2],'Color',ClustMap(kMeansClus(lidx),:),'LineWidth',1.5);
    %         plot([YL(lidx) YL(lidx)],[XL(lidx) XL(lidx)+AbsWin2],'Color',ClustMap(kMeansClus(lidx),:),'LineWidth',1.5);
    %         plot([YL(lidx) YL(lidx)+AbsWin2],[XL(lidx) XL(lidx)],'Color',ClustMap(kMeansClus(lidx),:),'LineWidth',1.5);
    %         plot([YL(lidx) YL(lidx)+AbsWin2],[XL(lidx)+AbsWin2 XL(lidx)+AbsWin2],'Color',ClustMap(kMeansClus(lidx),:),'LineWidth',1.5);
    %     end
        colormap(gca,SPoOFMap_colorcet('C5'));
        axis image;
        set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI');
        c = colorbar('Location','EastOutside','LineWidth',1.5,'Ticks',[-pi -pi/2 0 pi/2 pi],'TickLabels',{'-π','-π/2','0','π/2','π'});
        
    
        subplot(323); cla;
        hold all;
        imagesc( Clustered');
        colormap(gca,ClustMap);
        axis image;
        set(gca,'XTick',[],'YTick',[],'FontSize',15,'YDir','reverse','XColor',[1 1 1],'YColor',[1 1 1],'FontName','Segoe UI');
            c = colorbar('Location','EastOutside','LineWidth',1.5);
    
        %%
        CurrClustMed = zeros([length(taxis) NumClusters]);
        
        % figure(24); clf;
        subplot(122); 
        hold all;
        for idx = 1:NumClusters
        %     subplot(NumClusters,1,idx);
            count = 1;
            CurrClust = [];%zeros([numF 1]);
            for cidx = 1:N 
                if kMeansClus(cidx) == idx
                    CurrClust = [CurrClust T3CharMovMean(:,cidx)];
                    count = count + 1;
                    plot(taxis*1000, T3CharMovMean(:,cidx) + idx*10,'Color',[ClustMap(idx,:) 0.02],'LineWidth',1);
                end
            end
        %     CurrClust = CurrClust / count;
            CurrClustMed(:,idx) = mean(CurrClust,2);
        
            plot(taxis*1000, CurrClustMed(:,idx) + idx*10,'Color',ClustMap(idx,:),...
                'LineWidth',1);
            text(1000,idx*4+0.5,num2str(count),'FontSize',15,'FontName','Segoe UI')
            set(gca,'LineWidth',1.5,'FontSize',15,'FontName','Segoe UI','YDir','normal','XColor',[0 0 0],'YColor',[0 0 0]);
        
        end
        %
        subplot(325)
        CorrCoeff = zeros([NumClusters NumClusters]);
        
        for idx1 = 1:NumClusters
            for idx2 = 1:idx1
                cccc = corrcoef(CurrClustMed(:,idx1),CurrClustMed(:,idx2));
                CorrCoeff(idx1,idx2) = cccc(1,2);
            end
        end
        %
        imagesc(CorrCoeff);
        colormap(gca,SPoOFMap_colorcet('D5'));
        axis image;
        set(gca,'FontSize',15,'YDir','reverse','XColor',[0 0 0],'YColor',[0 0 0],'FontName','Segoe UI');
        c = colorbar('Location','EastOutside','LineWidth',1.5);
        clim([-1 1])
    %%
        CheckCond = sum( abs(CorrCoeff(:)) > 0.9) - NumClusters;
        disp(['Iteration ' num2str(iter) ' in ' num2str(toc,'%1.1f') ' s... (' num2str(CheckCond) ')'])
        iter = iter + 1;
        
    end
    
    %%
    savefig(gcf,[pwd '\' fbase '\' fbase '_kMeans_9NOV2023.fig']);%%

    %%

    figure(300);
    set(gcf,'Color',[0 0 0],'InvertHardCopy','off');

    FID1 = fopen([pwd '\' fbase '\' OutName1 '.bin'],'rb');
    FID2 = fopen([pwd '\' fbase '\' OutName2 '.bin'],'rb');
    
    FAvg = 10;
    numFrames = numF;
%     wBar = waitbar(0,'Starting...');
    tic;

    v = VideoWriter('HeartBeat.mp4','MPEG-4');
    v.FrameRate = 150;
    v.Quality = 60;

    open(v)
    for frameIdx = 1:(numFrames/FAvg)
        Out_P1 = fread(FID1,numx*numy*2 * FAvg,'float');
        Out_P1 = reshape( Out_P1(1:2:end-1) + 1i.*Out_P1(2:2:end),[numx numy FAvg]);
    
        Out_P2 = fread(FID2,numx*numy*2*FAvg,'float');
        Out_P2 = reshape( Out_P2(1:2:end-1) + 1i.*Out_P2(2:2:end),[numx numy FAvg]);
    
        Out_P1 = exp(-1i*PhaseMask1).*Out_P1;
        Out_P2 = exp(1i*PhaseMask2).*conj(Out_P2).*exp(1i.*-pi/2);

        delta_s = 180/pi*atan2( abs( 1i.*Out_P2 - Out_P1), abs( 1i.*Out_P2 + Out_P1));


        yshow = [21:numy];
        xshow = [1:(numx-20)];

        subplot(2,3,[1 2]);
        imagesc(mean( abs( [Out_P1(yShow,xShow,:) zeros([length(yshow) 10 FAvg]) Out_P2(yShow,xShow,:)]),3));
        text(10,12,[num2str(taxis((frameIdx-1)*FAvg + 1),'%1.3f') ' s'],'FontName','Segoe UI','FontSize',20,'Color','y','FontWeight','bold');
        axis image;
        set(gca,'XColor','k','YColor','k','XTick',[],'YTick',[],'FontName','Segoe UI','FontSize',15);
        colormap(gca,SPoOFMap_colorcet('L1'));
        c = colorbar('Location','EastOutside','LineWidth',1.5,'Color','w');
        c.Label.String = 'OCT intensity (a.u.)';
        clim([0 5000]);



        Back1x = 45:64;%1:numx;
        Back1y = 125:144;%1:numy;

        angP1 = median( angle( Out_P1(:,:,round(FAvg/2))),3);
        BackSub = median( reshape(angP1(Back1x,Back1y),[length(Back1x)*length(Back1y) 1]));
        angP1 = angle( exp(1i.*angP1).*exp(-1i.*BackSub).*exp(1i.*pi/1.5));
        angP2 = median( angle( Out_P2(:,:,round(FAvg/2))),3);
        angP2 = angle( exp(1i.*angP2).*exp(-1i.*BackSub).*exp(1i.*pi/1.5));
        subplot(2,3,[4 5]);
        imagesc([angP1(yShow,xShow) zeros([length(yshow) 10]) angP2(yShow,xShow)]);
%         text(20,20,[num2str(taxis(frameIdx),'%1.3f') ' s'],'FontName','Segoe UI','FontSize',15,'Color','y');
        axis image;
        set(gca,'XColor','k','YColor','k','XTick',[],'YTick',[],'FontName','Segoe UI','FontSize',15);
        colormap(gca,OCMPhaseMap);
        c = colorbar('Location','EastOutside','LineWidth',1.5,'Color','w');
        c.Label.String = 'OCT phase (rad)';
        clim([-pi pi]);

        subplot(2,3,3);
        imagesc(mean( abs( Out_P1(yShow,xShow)),3)./(mean( abs( Out_P1(yShow,xShow)),3) + mean( abs( Out_P2(yShow,xShow)),3)));
%         text(10,10,[num2str(taxis(frameIdx),'%1.3f') ' s'],'FontName','Segoe UI','FontSize',15,'Color','y');
        axis image;
        set(gca,'XColor','k','YColor','k','XTick',[],'YTick',[],'FontName','Segoe UI','FontSize',15);
        colormap(gca,SPoOFMap_colorcet('D1A'));
        c = colorbar('Location','EastOutside','LineWidth',1.5,'Color','w');
        c.Label.String = 'OCT intensity ratio';
        clim([0.25 0.75]);

        subplot(2,3,6);
        imagesc(mean( delta_s(yShow,xShow,round(FAvg/2)),3));
%         text(20,20,[num2str(taxis(frameIdx),'%1.3f') ' s'],'FontName','Segoe UI','FontSize',15,'Color','y');
        axis image;
        set(gca,'XColor','k','YColor','k','XTick',[],'YTick',[],'FontName','Segoe UI','FontSize',15);
        colormap(gca,SPoOFMap_colorcet('R2'));
        c = colorbar('Location','EastOutside','LineWidth',1.5,'Color','w');
        c.Label.String = 'Retardation angle (^o)';
        clim([0 90]);

%         pause(0.1);
        FrameCurr = getframe(gcf);
        writeVideo(v,FrameCurr.cdata(:,181:(end-20),:));
    end
%     close(wBar);
    fclose(FID2);
    fclose(FID1);
    close(v)



    
    %%
    
    save([pwd '\' fbase '\Results_' fbase '_9NOV2023.mat'],'T3_N','T3ChartNorm','TStd_norm','StdVids','taxis','Filt','T3CharMovMean','kMeansClus');

    disp(['Finished file ' fbase ' in ' num2str(toc(tStarttic)/60,'%1.0f') ' mins...'])
end