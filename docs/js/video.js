const videosList = [
    {
        video: 'https://www.youtube.com/embed/MX8g-8NXx7w',
        thumbnail: 'https://img.youtube.com/vi/MX8g-8NXx7w/maxresdefault.jpg',
        title: 'Red cars',
        prompt: `<p style="font-family: 'Courier New', monospace; font-weight: bold; text-align: center;">"Track only <span style="color: orange;">red color</span> <span style="font-weight: bold; color: green;">car</span>"
        </p>`
    },
    {
        video: 'https://www.youtube.com/embed/H7nVN4VRhbk',
        thumbnail: 'https://img.youtube.com/vi/H7nVN4VRhbk/maxresdefault.jpg',
        title: 'Yellow cars',
        prompt: `<p style="font-family: 'Courier New', monospace; font-weight: bold; text-align: center;">"Track only <span style="color: orange;">yellow color</span> <span style="font-weight: bold; color: green;">car</span>"
        </p>`
    },
    {
        video: 'https://www.youtube.com/embed/E7mSlziENQ4',
        thumbnail: 'https://img.youtube.com/vi/E7mSlziENQ4/maxresdefault.jpg',
        title: 'Dark blue cars',
        prompt: `<p style="font-family: 'Courier New', monospace; font-weight: bold; text-align: center;">"Track only <span style="color: orange;">dark blue color</span> <span style="font-weight: bold; color: green;">car</span>"
        </p>`
    },
    {
        video: 'https://www.youtube.com/embed/6Yte9nL2d0k',
        thumbnail: 'https://img.youtube.com/vi/6Yte9nL2d0k/maxresdefault.jpg',
        title: 'Airplanes',
        prompt: `<p style="font-family: 'Courier New', monospace; font-weight: bold; text-align: center;">"Track <span style="color: orange;">metal shell</span> <span style="font-weight: bold; color: green;">airplane</span>"
    </p>`
    },
    {
        video: 'https://www.youtube.com/embed/Eh5HJUSL9NE',
        thumbnail: 'https://img.youtube.com/vi/Eh5HJUSL9NE/maxresdefault.jpg',
        title: 'Helicopters',
        prompt: `<p style="font-family: 'Courier New', monospace; font-weight: bold; text-align: center;">"Track <span style="color: orange;">flying</span> <span style="font-weight: bold; color: green;">helicopter</span>"
    </p>`
    },
    ]
    
    const categories = [...new Set(videosList.map((item) => { return item }))]
    document.getElementById('videosList').innerHTML = categories.map((item) => {
        var { video, thumbnail, title, prompt } = item;
        return (
            `<div class="list active">
            <div class="list-prompt" hidden>${prompt}</div>
            <iframe src=${video} class="list-video" hidden></iframe>
            <img src=${thumbnail} class="list-video">
            <div style="width: 100%; height: 100%">
                <p class="list-title">${title}</p>
            </div>
            </div>`
            )
    }).join('')
    
    let videoList = document.querySelectorAll('.video-list-container .list');
    videoList.forEach(remove => { remove.classList.remove('active') });
    videoList.forEach(vid => {
        vid.onclick = () => {
            videoList.forEach(remove => { remove.classList.remove('active') });
            vid.classList.add('active');
            let src = vid.querySelector('.list-video').src;
            let title = vid.querySelector('.list-title').innerHTML;
            let prompt = vid.querySelector('.list-prompt').innerHTML;
            document.querySelector('.main-video-container .main-video').src = src;
            document.querySelector('.main-video-container .prompt').innerHTML = prompt;
        };
    });
    