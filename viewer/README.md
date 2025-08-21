# super-resolution-viewer

![](https://github.com/refracta/super-resolution-viewer/assets/58779799/d8f76bd3-421d-4c49-a473-cf8454696edc)
![](https://github.com/refracta/super-resolution-viewer/assets/58779799/b0f9c381-2ad9-4953-a7a8-8beee69c9295)

Image comparison viewers specialized in super-resolution and various vision-related tasks.

## Installation

### python webserver

```
cd [HOSTING_DIRECTORY]
python -m http.server [LISTEN_PORT]
```

### nginx (recommend)

```bash
docker run --name super-resolution-viewer -p [LISTEN_PORT]:80 -v [HOSTING_DIRECTORY]:/usr/share/nginx/html -v [THIS_PROJECT_ROOT]/deploy/default.conf:/etc/nginx/conf.d/default.conf -d nginx:stable-alpine3.17-slim
# example
cd super-resolution-viewer
docker run --name super-resolution-viewer -p 8080:80 -v $(pwd):/usr/share/nginx/html -v $(pwd)/deploy/default.conf:/etc/nginx/conf.d/default.conf -d nginx:stable-alpine3.17-slim
```

### github.io

- You can deploy this project file to the `gh-pages` branch for use.
- If hosted on github.io, it automatically recognizes this and uses the git tree API.
- If necessary, you can set the deployment branch in [js/github.io.json](js/github.io.json).

## Quickstart

- `configs` 경로에 비교할 이미지 대상 목록을 포함하는 설정 파일을 구성한다.

### [configs/demos.json](configs/demos.json)

```json
{
  "title": "demos",
  "targets": [
    {
      "groundTruth": true,
      "path": "demos/{data}/a"
    },
    {
      "path": "demos/{data}/b"
    },
    {
      "path": "demos/{data}/c"
    },
    {
      "path": "demos/{data}/d"
    }
  ]
}
```

- `targets`: 각 경로 정보 항목의 리스트 [1개 항목 이상 필수]
- `targets[?]`: 경로 정보 항목
- `targets[?].path`: 이미지가 저장된 폴더 경로, {GET_PARAMETER_NAME}의 형태로 GET 매개변수 값을 경로에 사용할 수 있다. [필수]
- `targets[?].groundTruth`: 타겟이 기준 데이터(groundTruth)인지의 여부를 설정
  해당 설정 파일을 구성 후, 뷰어 페이지에 GET 매개변수로 `config=demos.json`을 설정하여 뷰어를 사용할 수 있다.

- [?config=demos.json&data=alpha](https://refracta.github.io/super-resolution-viewer?config=demos.json&data=alpha)
- [?config=demos.json&data=beta](https://refracta.github.io/super-resolution-viewer?config=demos.json&data=beta)

### Advanced configuration

뷰어의 거의 모든 매개변수는 설정 파일 `config/**/*.json`과 `GET 매개변수`를 통해 동시에 설정할 수 있다.
`GET 매개변수`의 설정값이, `config/**/*.json`의 설정값보다 높은 우선순위를 가진다.

- config: `configs/**` 내부에 위치한 설정 파일의 이름을 지정한다. [GET] (String, Required)
- configPath: 설정 파일의 경로를 수동으로 지정한다. `configs/**` 이 옵션을 통해 내부에 위치하지 않은 설정 파일을 사용할 수 있다. [GET] (String)
- type: 뷰어의 경로 [맵핑 유형](#mappers)을 설정한다. 미설정 시 기본 매핑 로직이 적용된다. [GET, JSON] (String, Default="default")

```
[GET]
?type=mapping-type-name

[JSON]
{
  "type": "mapping-type-name"
}
```

- title: 뷰어의 제목을 설정한다. 이 제목은 브라우저 탭이나 페이지 제목에 사용된다. [GET, JSON] (String, Default="Super-Resolution Viewer")
- indexes: 특정 이미지 파일들의 색인을 설정한다. 본 설정이 적용되면 뷰어는 해당 인덱스 목록만 순회하게 된다. [GET, JSON] (Integer Array,
  Default=[All of Image Indexes])

```
[GET]
?indexes=1.3.5 (seperator='.', ',', '*')

[JSON]
{
  "indexes": [1, 3, 5, 7, 8]
}
 ```

- index: 초기에 보여질 이미지의 색인을 설정한다. 이 값은 사용자가 이미지를 넘길 때 업데이트 된다. [GET, JSON] (Integer, Default=0)
- preloadSize: 현재 색인을 기준으로 이전 다음 이미지를 몇 개씩 미리 로딩할 지 설정한다. [GET, JSON] (Integer, Default=3)
- hides: 특정 타겟을 숨기기 위한 색인 배열을 설정한다. 특정 이미지들을 뷰에서 제외시킬 때 사용된다. targets[?].hide와 동시에 적용된다. [GET, JSON] (Integer Array,
  Default=[])
- canvasLeftColor: 그리기 도구에서 사용할 왼쪽 버튼의 색상을 설정한다. [GET, JSON] (CSS Color String)

```
[GET]
?canvasLeftColor=yellow
?canvasLeftColor=ff00ff
?canvasLeftColor=rgba(100,100,100,0.5)

[JSON]
{
  "canvasLeftColor": "ff00ff"
}

 ```

- canvasRightColor: 그리기 도구에서 사용할 오른쪽 버튼의 색상을 설정한다. [GET, JSON] (CSS Color String)
- canvasThickness: 그리기 도구의 선 두께를 설정한다. [GET, JSON] (Integer, Unit:px, Default=3)
- SSIMWindowSize: SSIM 계산 시 사용할 윈도우 크기를 설정한다. [GET, JSON] (Integer, Unit:px, Default=11)
- PSNRGridWidth: PSNR 시각화 그리드의 너비를 설정한다. [GET, JSON] (Integer, Unit:px, Default=5)
- PSNRGridHeight: PSNR 시각화 그리드의 높이를 설정한다. [GET, JSON] (Integer, Unit:px, Default=5)
- PSNRGridSize: PSNR 그리드의 크기를 설정한다. 이 값이 설정되면, 너비와 높이가 이 크기로 설정된다. [GET, JSON] (Integer, Unit:px, Default=null)
- pageZoom: 현재 페이지의 줌 수준을 강제로 지정한다. [GET, JSON] (Float, Default=1)
- pageZoomDelta: 페이지 줌 조정 시의 증감 단위를 설정한다. 이 값은 마우스 휠 이벤트에 따라 페이지 줌 레벨을 조정할 때 사용된다. [GET, JSON] (Float, Default=0.01 (
  1%))
- showingPSNRVisualizer: PSNR 시각화 도구의 활성화 여부를 설정한다. [GET, JSON] (Boolean, Default=false)
- diffIndex: 차이 이미지를 보여줄 때 기준이 될 이미지 색인을 설정한다. [GET, JSON] (Integer, Default=-1)
- zoomMode: 줌 모드의 활성화 여부를 설정한다. 이 모드가 활성화되면 사용자는 특정 영역을 확대하여 볼 수 있다. [GET, JSON] (Boolean, Default=false)
- zoomAreaWidthRatio: 줌 모드에서 확대할 영역의 너비 비율을 설정한다. [GET, JSON] (Float, Range: 0~1, Default=0.8)
- zoomAreaHeightRatio: 줌 모드에서 확대할 영역의 높이 비율을 설정한다. [GET, JSON] (Float, Range: 0~1, Default=0.8)
- zoomAreaWidth: 줌 영역의 실제 너비를 설정한다. [GET, JSON] (Integer, Default=100)
- zoomAreaHeight: 줌 영역의 실제 높이를 설정한다. [GET, JSON] (Integer, Default=100)
- zoomAreaColor: 줌 영역의 테두리 색상을 설정한다. [GET, JSON] (CSS Color String, Default="label" (targets[?].labelBackgroundColor))
- zoomAlpha: 줌 영역의 투명도를 설정한다. 이 값은 줌 영역이 그려질 때 사용된다. [GET, JSON] (Float, Range: 0~1, Default=0.5)
- zoomAreaThickness: 줌 영역의 테두리 두께를 설정한다. [GET, JSON] (Integer, Default=5)
- zoomAreaDelta: 줌 영역의 크기 조정 단위를 설정한다. 이 값은 마우스 휠을 사용하여 줌 영역의 크기를 조절할 때 사용된다. [GET, JSON] (Integer, Default=5)
- crop: 이미지를 잘라내어 다운로드할 때 사용할 초기 크롭 설정을 문자열로 지정한다. [GET, JSON] (CropFormat String, Default=null)

```
[GET]
?crop=x100y100w200h200 (Start at (100, 100), 200 x 200 Size)
?crop=x100y100w100h100d1 (diffIndex=1)
?crop=x100y100w100h100p1 (showingPSNRVisualizer=true)

[JSON]
{
  "crop": "crop=x100y100w100h100d1p1"
}
```

- configHelp: 설정 관련 도움말을 제공하는 텍스트이다. 뷰어 로딩에 실패하거나, 'F2' 또는 'w' 키를 눌렀을 때 보여진다. [GET, JSON] (String, Default=null)
- targets[?].path: 각 타겟에 대한 데이터 파일이나 리소스의 서버 경로를 지정한다. 이 경로는 뷰어가 필요한 리소스를 로드할 때 사용된다. `{paramName}` 형태의 문자열을 포함하는
  경우, `GET 매개변수`에 `paramName` 매개변수가 존재하는 경우 해당 값으로 교체되어 설정된다. 중괄호를 사용하여야 할 경우 `"\\{data\\}"`의 형태로 이스케이핑할 수 있다. [JSON] (
  String, Required)
- targets[?].label: 각 타겟을 식별하기 위한 레이블 텍스트를 설정한다. 이 레이블은 사용자 인터페이스에 표시되어 타겟을 구분하는 데 사용된다. [JSON] (String,
  Default=targets[?].path)
- targets[?].labelColor: 타겟 레이블의 텍스트 색상을 설정한다. 사용자 인터페이스에서 레이블을 더욱 눈에 띄게 할 수 있다. [JSON] (CSS Color String,
  Default=hashColor(targets[?].path))
- targets[?].labelBackgroundColor: 타겟 레이블의 배경 색상을 설정한다. 이 색상은 레이블의 가독성을 향상시키는 데 도움을 준다. [JSON] (CSS Color String,
  Default: whiteOrBlack(targets[?].labelColor))
- targets[?].hide: 특정 타겟을 사용자 인터페이스에서 숨길지 결정한다. 이 값이 true로 설정되면 해당 타겟은 뷰어에 표시되지 않는다. [JSON] (Boolean, Default=false)
- targets[?].ignore: 특정 타겟을 전체 처리 로직에서 완전히 무시한다. [JSON] (Boolean, Default=false)
- targets[?].groundTruth: 타겟이 기준 데이터(groundTruth)인지의 여부를 설정한다. 설정한 타겟의 이미지 목록이 전체의 기준으로 설정된다. `type=results`를 사용시 필수적으로
  하나의 타겟에 설정해야 하며, PSNR, SSIM 계산 등에 해당 타겟이 기준으로 사용된다. (이 값이 설정되지 않으면 뷰어는 첫번째 타겟의 이미지 목록을 기준으로 사용한다) [JSON] (Boolean,
  Default=false)
- targets[?].files: 타겟별로 내부 이미지 파일 목록이 다른 경우, 존재하는 내부 이미지 파일 목록을 수동으로 구성할 때 사용한다. (FileName String Array, Default=[files
  of targets[?].path])

## mappers

`js/mappers.js`에 경로 맵핑 함수를 직접 설정하여 사용자 지정 경로 맵핑을 처리할 수 있다.

### [js/mappers.js](js/mappers.js)

```js
export default {
  "mapping-type-name": {
    // type 설정과 동일한 이름으로 설정
    file: (target, file, viewer) => `${target.path}/${file}`,
    // target, file (기준 파일 목록의 파일), viewer (모든 설정 속성을 프로퍼티로 가지고 있는 객체)
    // 주어진 매개변수로부터 최종 이미지 경로를 반환하게 작성한다.
    targetBefore: (target, index, array, viewer) => target,
    // target.files를 가져오기 위해, target.path에 HTTP GET 요청을 전송하기 직전에 호출된다.
    // 주어진 매개변수로부터 수정한 target 객체를 반환한다.
    targetAfter: (target, index, array, viewer) => target
    // target.files가 설정되고, target 관련 설정이 최종적으로 완료된 후에 호출된다.
    // 주어진 매개변수로부터 수정한 target 객체를 반환한다.
  }, ...
}
```

## Shortcut

```
F1, q: help
F2, f: config help
← , →: previous, next image
n , m: previous, next image
(i)ndex: move to index
(r)eset
shift + wheel: page zoom

(a)dd to favorites
(l)ist favorites (toggle)
(d)elete favorites
F3: delete all favorites

(p)snr visualizer (toggle)
1 ~ 9: show image diff (toggle)
(c)apture page

(z)oom mode (toggle)
wheel: resize zoom area (in zoom mode)
(w) + wheel: resize zoom area width (in zoom mode)
(h) + wheel: resize zoom area height (in zoom mode)
space: download cropped images (in zoom mode)
shift + space: download cropped images with original images (in zoom mode)
(u)rl: copy crop url (in zoom mode)
```
