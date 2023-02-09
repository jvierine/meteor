;maarsy_antenna_pattern, antenna=1,max_phi = 90
PRO MAARSY_ANTENNA_PATTERN,antenna=antenna,phasetype = phasetype,graph=graph,$
  max_phi = max_phi,dcosx=dcosx,dcosy=dcosy, power=power, two_way = two_way,overplot=overplot,fill=fill,$
  az0=az0,zenith0=zenith0,xrange=xrange,yrange=yrange,$
  pointing_phases = pointing_phases,$
  pointing_labels = pointing_labels,talk = talk, damaged=damaged,mincol = mincol,$
  dcoscut = dcoscut,newpar=newpar,newpp=newpp, write_ascii= write_ascii

IF N_ELEMENTS(write_ascii) EQ 0 THEN write_ascii = 0    ; 0: no, 1: Yes, antenna pattern
IF N_ELEMENTS(dcoscut) EQ 0 THEN dcoscut = 0
IF N_ELEMENTS(mincol) EQ 0 THEN mincol = 50.
IF N_ELEMENTS(damaged) EQ 0 THEN damaged = 0
IF N_ELEMENTS(talk) EQ 0 THEN talk = 1
IF N_ELEMENTS(antenna) EQ 0 THEN antenna = 0  ;0: All, 1: Anemony, 2: Hexagon
IF N_ELEMENTS(phasetype) EQ 0 THEN phasetype = 0
IF N_ELEMENTS(max_phi) EQ 0 THEN max_phi = 15
IF N_ELEMENTS(two_way) EQ 0 THEN two_way = 0  ; 0: One way, 1: Two way
IF N_ELEMENTS(graph) EQ 0 THEN graph = 1
IF N_ELEMENTS(overplot) EQ 0 THEN overplot = 0
IF N_ELEMENTS(az0) EQ 0 THEN az0 = 0  ; pointing azimuth with respect to N (degrees)
IF N_ELEMENTS(zenith0) EQ 0 THEN zenith0 = 0  ; pointing zenith (degrees)


maxdcos = SIN(max_phi*!PI/180)

IF N_ELEMENTS(dcosx) EQ 0 THEN BEGIN
    nx = 201 ;101
    dcosx = (FINDGEN(nx)/(nx-1)-0.5)*2*maxdcos
ENDIF
IF N_ELEMENTS(dcosy) EQ 0 THEN BEGIN
    ny = 201 ;101
    dcosy = (FINDGEN(ny)/(ny-1)-0.5)*2*maxdcos
ENDIF
nx = N_ELEMENTS(dcosx)
ny = N_ELEMENTS(dcosy)
maxdcos = MAX(dcosx) > MAX(dcosy)

IF N_ELEMENTS(xrange) EQ 0 THEN xrange = maxdcos*[-1,1]
IF N_ELEMENTS(yrange) EQ 0 THEN yrange = maxdcos*[-1,1]

; Constants

lambda = 300./53.5 ;
kk =2*!PI/lambda
jj = COMPLEX(0,1)



section_centers = [[28.,15.],[28,75],[28.,135.],[28, -165.],[28.,-105.],[28., -45.]]
center_label = ['A','B','C','D','E','F']

element = [[4.,-45],[4.,-105],[0.,0.],[4.,-165],[4.,135.],[4.,75],[4.,15]]

element_label = STRING(INDGEN(7)+1, FORMAT = '(I1)')

hexagons = [[10.58,-4.1],[10.58,-64.1],[0.0,0.0],[10.58,-124.1],[10.58,175.9],[10.58,115.9],[10.58,55.9],[18.33, 85.9], [18.33,-154.1]]

hexagon_label = STRING(INDGEN(9)+1, FORMAT = '(I2.2)')

hexagon_rotation = [-120,+120,00,180.,-60,60.0,180,-60,60.]

; Elements of an Hexagon
nel = N_ELEMENTS(element_label)
; Hexagons per section
nhex = N_ELEMENTS(hexagon_rotation)
; Number of sections
nsection = N_ELEMENTS(section_centers(0,*))

non_hex_el_xy = [$
  [5*4.*SIN(-45*!PI/180),5*4.*COS(-45*!PI/180)],$
  [4*4.*SIN(-45*!PI/180),4*4.*COS(-45*!PI/180)],$
  [4*4.*SIN(-45*!PI/180)+1*4*SIN(15*!PI/180),4*4.*COS(-45*!PI/180)+1*4*COS(15*!PI/180)],$
  [3*4.*SIN(-45*!PI/180)+1*4*SIN(15*!PI/180),3*4.*COS(-45*!PI/180)+1*4*COS(15*!PI/180)],$
  [3*4.*SIN(-45*!PI/180)+2*4*SIN(15*!PI/180),3*4.*COS(-45*!PI/180)+2*4*COS(15*!PI/180)],$
  [4*4.*SIN(75*!PI/180)+1*4*SIN(15*!PI/180),4*4.*COS(75*!PI/180)+1*4*COS(15*!PI/180)],$
  [3*4.*SIN(75*!PI/180)+2*4*SIN(15*!PI/180),3*4.*COS(75*!PI/180)+2*4*COS(15*!PI/180)],$
  [3*4.*SIN(15*!PI/180)+1*4*SIN(75*!PI/180),3*4.*COS(15*!PI/180)+1*4*COS(75*!PI/180)]]
non_hex_label = STRING(INDGEN(8)+1,FORMAT = '(I1)')  

non_hex_el = TRANSPOSE([[SQRT(TOTAL(non_hex_el_xy^2,1))],[REFORM(ATAN(non_hex_el_xy(0,*),non_hex_el_xy(1,*)))*180/!PI]])

n_nonhex = N_ELEMENTS(non_hex_el_xy(0,*))  


nant = nel*nhex*nsection+n_nonhex*nsection+nel
all_section_xy = FLTARR(2,nant)
all_section_label = STRARR(nant)

el_index = INDGEN(nel)
non_hex_index = INDGEN(n_nonhex)

FOR is = 0l, nsection-1l DO BEGIN 
  section_rotation = (section_centers(1,is)-section_centers(1,0))
  section_offset = section_centers(0,is)*[SIN(section_centers(1,is)*!PI/180),COS(section_centers(1,is)*!PI/180)]
  FOR ih = 0l, nhex-1l DO BEGIN
    iptr = el_index+ih*nel+is*nhex*nel      
    rot_element = (element(1,*)+hexagon_rotation(ih)+section_rotation)*!PI/180
    rot_hexagon = (hexagons(1,ih)+section_rotation)*!PI/180
    hexagon_offset = hexagons(0,ih)*[SIN(rot_hexagon),COS(rot_hexagon)]
    all_section_xy(*,iptr) = $
; XY of each element
        [element(0,*)*SIN(rot_element),element(0,*)*COS(rot_element)]+$
        hexagon_offset # REPLICATE(1,nel)+$
        section_offset # REPLICATE(1,nel)  
    all_section_label(iptr) = center_label(is)+'-'+hexagon_label(ih)+'_'+element_label(*)  
  ENDFOR
; Filling vectors for non hexagonal elements  
  nptr = nel*nhex*nsection+non_hex_index+is*n_nonhex
  rot_element = (non_hex_el(1,*)+section_rotation)*!PI/180
  all_section_xy(*,nptr) = $
; XY of each element
        [non_hex_el(0,*)*SIN(rot_element),non_hex_el(0,*)*COS(rot_element)]+$
        section_offset # REPLICATE(1,n_nonhex)  
    all_section_label(nptr) = center_label(is)+'-10'+'_'+non_hex_label(*)  
ENDFOR
; Filling the center hexagon
cptr = el_index+nel*nhex*nsection+n_nonhex*nsection
all_section_xy(*,cptr) =  [element(0,*)*SIN(element(1,*)*!PI/180),$
                           element(0,*)*COS(element(1,*)*!PI/180)]
all_section_label(cptr) = 'F-11'+'_'+element_label(*)

; Just section and hexagon number
jlabel = STRMID(all_section_label,0,4)
CASE antenna OF
0: BEGIN    ; All
  valid = INDGEN(nant)
  pos = all_section_xy
  tantenna = 'All'
  
; Printing phases of Anemony centers
;   
  clabel = STRMID(all_section_label,1,5)
  center_valid = WHERE(clabel EQ '-03_3' OR clabel EQ '-11_3',ccvalid)
  IF talk NE 0 THEN PRINT,all_section_label(center_valid)

; Printing phases of Hexagon centers used in Meteor experiment (09/11/2012)

  my_hexagons = ['A-07','B-04', 'C-05',  'D-04', 'E-07', 'F-04', 'D-09', 'C-08']
  nhexcenter = N_ELEMENTS(my_hexagons)
  hexagon_valid = INTARR(nhexcenter)
  FOR ih = 0, nhexcenter-1 DO BEGIN
    hexagon_valid(ih) = WHERE(all_section_label EQ my_hexagons(ih)+'_3')
  ENDFOR  
  IF talk NE 0 THEN PRINT,all_section_label(hexagon_valid)
END
1: BEGIN    ; Anemony  
  valid = WHERE(jlabel EQ 'A-09' OR $
    jlabel EQ 'B-09' OR $
    jlabel EQ 'C-09' OR $
    jlabel EQ 'D-09' OR $
    jlabel EQ 'E-09' OR $
    jlabel EQ 'F-09' OR $
    jlabel EQ 'F-11')
  pos = all_section_xy(*,valid)
  tantenna = 'Anemony'
END
2: BEGIN    ; Hexagon
  valid = WHERE(jlabel EQ 'F-11')
  pos = all_section_xy(*,valid)
  tantenna = 'Hexagon'
END
3: BEGIN    ; All the anemonies
  novalid = WHERE(STRMID(jlabel,1,3) EQ '-10' OR STRMID(jlabel,1,3) EQ '-08',cnovalid,COMPLEMENT=valid,NCOMPLEMENT = cvalid)
  pos = all_section_xy(*,valid)
  tantenna = 'All Anemonies'
END
4: BEGIN      ; Circular RC ACE
  valid = WHERE($
    jlabel EQ 'A-01' OR $
    jlabel EQ 'A-02' OR $
    jlabel EQ 'A-03' OR $
    jlabel EQ 'A-04' OR $
    jlabel EQ 'A-05' OR $
    jlabel EQ 'A-06' OR $
    jlabel EQ 'A-07' OR  $
    jlabel EQ 'C-01' OR $
    jlabel EQ 'C-02' OR $
    jlabel EQ 'C-03' OR $
    jlabel EQ 'C-04' OR $
    jlabel EQ 'C-05' OR $
    jlabel EQ 'C-06' OR $
    jlabel EQ 'C-07' OR $
    jlabel EQ 'E-01' OR $
    jlabel EQ 'E-02' OR $
    jlabel EQ 'E-03' OR $
    jlabel EQ 'E-04' OR $
    jlabel EQ 'E-05' OR $
    jlabel EQ 'E-06' OR $
    jlabel EQ 'E-07' OR $
    jlabel EQ 'A-09' OR $
    jlabel EQ 'C-09' OR $
    jlabel EQ 'E-09' OR $
    jlabel EQ 'A-08' OR $
    jlabel EQ 'C-08' OR $
    jlabel EQ 'E-08', COMPLEMENT = novalid)
  pos = all_section_xy(*,valid)  
;  pos = all_section_xy(*,novalid)
  tantenna = 'ACE'
END
5: BEGIN      ; Circular LC BDF
  valid = WHERE(jlabel EQ 'B-01' OR $
    jlabel EQ 'B-02' OR $
    jlabel EQ 'B-03' OR $
    jlabel EQ 'B-04' OR $
    jlabel EQ 'B-05' OR $
    jlabel EQ 'B-06' OR $
    jlabel EQ 'B-07' OR  $
    jlabel EQ 'D-01' OR $
    jlabel EQ 'D-02' OR $
    jlabel EQ 'D-03' OR $
    jlabel EQ 'D-04' OR $
    jlabel EQ 'D-05' OR $
    jlabel EQ 'D-06' OR $
    jlabel EQ 'D-07' OR $
    jlabel EQ 'F-01' OR $
    jlabel EQ 'F-02' OR $
    jlabel EQ 'F-03' OR $
    jlabel EQ 'F-04' OR $
    jlabel EQ 'F-05' OR $
    jlabel EQ 'F-06' OR $
    jlabel EQ 'F-07' OR $
    jlabel EQ 'B-09' OR $
    jlabel EQ 'D-09' OR $
    jlabel EQ 'F-09' OR $
;    jlabel EQ 'F-11' OR $   ; Array center
    jlabel EQ 'B-08' OR $
    jlabel EQ 'D-08' OR $
    jlabel EQ 'F-08', COMPLEMENT = novalid)
  pos = all_section_xy(*,valid)
  tantenna = 'BDF'
END  
10: BEGIN    ; Two Anemony  
  valid = WHERE(jlabel EQ 'A-01' OR $
    jlabel EQ 'A-02' OR $
    jlabel EQ 'A-03' OR $
    jlabel EQ 'A-04' OR $
    jlabel EQ 'A-05' OR $
    jlabel EQ 'A-06' OR $
    jlabel EQ 'A-07' OR  $
    jlabel EQ 'D-01' OR $
    jlabel EQ 'D-02' OR $
    jlabel EQ 'D-03' OR $
    jlabel EQ 'D-04' OR $
    jlabel EQ 'D-05' OR $
    jlabel EQ 'D-06' OR $
    jlabel EQ 'D-07')
  pos = all_section_xy(*,valid)
  tantenna = 'Anemony'
END
ENDCASE

IF 2 EQ 1 THEN BEGIN
;  IF graph EQ 1 OR graph GE 10 THEN WINDOW,2, XSIZE = 700, YSIZE = 700
  figfilen = 'conf_'+ tantenna
  CONTROL_PLOT,pp =graph(0), CLOSE = 0, FILEN = 'ante_figs/'+figfilen
  
  PLOT,[-50,50],[-50,50],/NODATA
  FOR i = 0, nant-1 DO XYOUTS,all_section_xy(0,i),all_section_xy(1,i),all_section_label(i),ALIGN = 0.5,/DATA
  FOR iv = 0, N_ELEMENTS(valid)-1 DO $
      XYOUTS,all_section_xy(0,valid(iv)),all_section_xy(1,valid(iv)),all_section_label(valid(iv)),ALIGN = 0.5,/DATA,COLOR=1
  CONTROL_PLOT,pp =graph(0), CLOSE = 1, FILEN = 'ante_figs/'+figfilen

ENDIF

CASE phasetype OF
0: BEGIN
  phase = -kk*(pos(0,*)*SIN(az0*!PI/180)*SIN(zenith0*!PI/180)+pos(1,*)*COS(az0*!PI/180)*SIN(zenith0*!PI/180))
  gain = phase*0.0+1.0
;  gain = gain+0.2*RANDOMN(seed, N_ELEMENTS(phase))
;  phase = phase+0.3*RANDOMN(seed, N_ELEMENTS(phase))
  
  tphase = 'Linear Phase'
  array_center = [0.,0.]
  dd = SQRT((pos(0,*)-array_center(0))^2+(pos(1,*)-array_center(1))^2)
  alpha = 0.5
; gain = alpha-(alpha-1.)*COS(!PI*dd/MAX(dd))

   
END
1: BEGIN    ; Chirp, parabolic phase front, with constant amplitude
    ph0y = 0.005  ;5.e-2
    ph0x = ph0y
    array_center = [0.,0.]
    phase = kk*(ph0x*(pos(0,*)-array_center(0))^2+ph0y*(pos(1,*)-array_center(1))^2)
    phase += -kk*(pos(0,*)*SIN(az0*!PI/180)*SIN(zenith0*!PI/180)+pos(1,*)*COS(az0*!PI/180)*SIN(zenith0*!PI/180))

    gain = phase*0.0 +1.0
    tphase = 'Parabolic + Amp const'
END
2: BEGIN    ; Chirp, parabolic phase front, with non constant amplitude
    ph0y = 0.005  ;5.e-2
    ph0x = ph0y
    array_center = [0.,0.]    
    phase = kk*(ph0x*(pos(0,*)-array_center(0))^2+ph0y*(pos(1,*)-array_center(1))^2)
    phase += -kk*(pos(0,*)*SIN(az0*!PI/180)*SIN(zenith0*!PI/180)+pos(1,*)*COS(az0*!PI/180)*SIN(zenith0*!PI/180))

    gain = phase*0.0 +1.0
    dd = SQRT((pos(0,*)-array_center(0))^2+(pos(1,*)-array_center(1))^2)
    alpha = 0.5
    gain = alpha-(alpha-1.)*COS(!PI*dd/MAX(dd))
   
    tphase = 'Parabolic + Amp tapering'
END
3: BEGIN      ; Random gain
    phase = -kk*(pos(0,*)*SIN(az0*!PI/180)*SIN(zenith0*!PI/180)+pos(1,*)*COS(az0*!PI/180)*SIN(zenith0*!PI/180))
    gain = phase*0.0+1.0- RANDOMU(seed,N_ELEMENTS(phase))*1.0
    tphase = 'Random amplitude'
END  
4: BEGIN    ; Chirp, parabolic phase front, with constant amplitude
  ph0y = 0.002  ;5.e-2
  ph0x = ph0y
  array_center = [0.,0.]
  phase = kk*(ph0x*(pos(0,*)-array_center(0))^2+ph0y*(pos(1,*)-array_center(1))^2)
  phase += -kk*(pos(0,*)*SIN(az0*!PI/180)*SIN(zenith0*!PI/180)+pos(1,*)*COS(az0*!PI/180)*SIN(zenith0*!PI/180))

  gain = phase*0.0 +1.0
  tphase = 'Parabolic + Amp const'
END
ENDCASE

IF N_ELEMENTS(ccvalid) GT 0 THEN BEGIN
  pointing_phases = FLTARR(ccvalid+nhexcenter)
  pointing_labels = STRARR(ccvalid+nhexcenter)
  pointing_phases(0:ccvalid-1) = phase(center_valid(*))
  pointing_labels(0:ccvalid-1) = all_section_label(center_valid(*))
  pointing_phases(ccvalid:*) = phase(hexagon_valid(*))
  pointing_labels(ccvalid:*) = all_section_label(hexagon_valid(*))
  IF talk NE 0 THEN BEGIN
    FOR ic = 0, ccvalid-1 DO BEGIN
      PRINT,all_section_label(center_valid(ic)), phase(center_valid(ic)) ;,all_section_xy(0,center_valid(ic)),all_section_xy(1,center_valid(ic))
    ENDFOR
    FOR ih = 0, nhexcenter-1 DO BEGIN
      PRINT,all_section_label(hexagon_valid(ih)), phase(hexagon_valid(ih)) ;,all_section_xy(0,hexagon_valid(ih)),all_section_xy(1,hexagon_valid(ih))
    ENDFOR
  ENDIF    
ENDIF

IF damaged GT 0 THEN BEGIN
  el_damaged = FIX(RANDOMU(seed, damaged)*N_ELEMENTS(gain))
  gain(el_damaged) = 0.0
ENDIF

IF dcoscut EQ 0 THEN BEGIN
  ee = COMPLEXARR(nx,ny)
  FOR iy=0,ny-1 DO $
    FOR ix = 0,nx-1 DO $
      ee(ix,iy) = TOTAL(gain*EXP(jj*kk*($
        pos(0,*)*dcosx(ix)+$
        pos(1,*)*dcosy(iy))+jj*phase))
ENDIF ELSE BEGIN
  ee = COMPLEXARR(nx)
  FOR ix = 0, nx-1 DO $
    ee(ix) = TOTAL(gain*EXP(jj*kk*($
      pos(0,*)*dcosx(ix)+$
      pos(1,*)*dcosy(ix))+jj*phase))
ENDELSE

power = ABS(ee)^2
amp = ABS(power)/MAX(ABS(power))

IF two_way EQ 1 THEN amp = amp^2
;PRINT,10*ALOG10(MAX(power))

IF graph NE 0 and dcoscut EQ 0 THEN BEGIN

  miny = 1.e-6
  ;lev = [0.,1.e-4,0.5*1.e-3,1.e-3,0.5*1.e-2,1.e-2,0.5*1.e-1,1.e-1,0.5,1]
  lev = [0.,1.e-4,1.e-3,1.e-2,1.e-1,0.25,0.5,1]
  ;lev = [0.,1.e-6,1.e-4,1.e-3,1.e-2,1.e-1,0.5,1]
  ;lev = [0.,1.e-3,1.e-2,1.e-1,0.5,1]                       ; For paper
  labels = STRING(10*ALOG10(lev),FORMAT='(I4)')
  ;col = FINDGEN(N_ELEMENTS(lev))/(N_ELEMENTS(lev)-1)*29+50.
  col = FINDGEN(N_ELEMENTS(lev))/(N_ELEMENTS(lev)-1)*29+mincol
  
  two_way_label = [ '','Two_way']
  
  IF overplot THEN junktitle = '' $
  ELSE junktitle = 'MAARSY Pattern: '+tantenna+' '+two_way_label(two_way)+' '+tphase
  
  figfilen = tantenna+two_way_label(two_way)+tphase
  
  ;!P.MULTI = [0,1,1]
  IF overplot EQ 1 THEN BEGIN
  ;  !P.MULTI(0) = 1
    xtitle = ' '
    ytitle = ' '
  
  ;  WSET,0 
  ENDIF ELSE BEGIN
    xtitle = 'X'
    ytitle = 'Y'
  
  ;  IF graph EQ 1 OR graph GE 10 THEN WINDOW,0, XSIZE = 700, YSIZE = 700
    CONTROL_PLOT,pp =graph(0), CLOSE = 0, FILEN = figfilen,$
      newpar=newpar,newpp=mewpp,/ENCAPSULATED
  ENDELSE
  
  CONTOUR,amp,dcosx,dcosy,OVERPLOT=overplot,$
         LEVELS=lev,C_COLORS=col,FILL=fill,$
         C_ANNOTATION = labels,$
         TITLE=junktitle,$
         YTITLE=xtitle,$
         XTITLE=ytitle,$
  ;     XGRIDSTYLE=1,XTICKLEN=0.5,$
  ;     YGRIDSTYLE=1,YTICKLEN=0.5,$
         XRANGE = xrange,YRANGE=yrange,$
         XSTYLE=1,YSTYLE=1,$
         CHARSIZE = charsize,NOCLIP=0
         
  IF overplot EQ 0 THEN CONTROL_PLOT,pp =graph(0), CLOSE = 1, FILEN = figfilen,$
      newpar=newpar,newpp=mewpp,/ENCAPSULATED


  IF write_ascii EQ 1 THEN BEGIN
    ascii_filen = '/Users/jchau/junk/maarsy_antenna_'+STRING(antenna,FORMAT = '(I2.2)')+'.txt'
    nfile = 12
    CLOSE, nfile
    OPENW,nfile,ascii_filen
    PRINTF,nfile,'MAARSY Antenna Pattern for ', tantenna
    PRINTF,nfile, 'Dcosx/Dcosy ', dcosx, FORMAT = '(A12,'+STRING(N_ELEMENTS(dcosx), FORMAT='(I3.3)')+'F12.3)' 
    FOR iy = 0, N_ELEMENTS(dcosy)-1 DO BEGIN
      PRINTF,nfile, dcosy(iy), amp(*,iy), FORMAT = '(F12.3,'+STRING(N_ELEMENTS(dcosx), FORMAT='(I3.3)')+'F12.3)' 
    ENDFOR
    CLOSE,nfile
  ENDIF
ENDIF

END