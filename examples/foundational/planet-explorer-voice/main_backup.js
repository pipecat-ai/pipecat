const cesiumToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyNmZmMDhhMy0wNTI1LTRhZjUtODdkNi1hYjdiMmYzMWZhNWMiLCJpZCI6MzAxMDM5LCJpYXQiOjE3NDY3OTk5MTl9.HOjPZnRwoypLqSTXmCYp2vn0ValjyKcrh3t3VHsKlbo';
const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('canvas');
const canvasCtx = canvasElement.getContext('2d');
let viewer, camera, infoBox, hoveredEntity;

// Gesture variables with safer defaults
let currentZoomLevel = 1.0;
let targetZoomLevel = 1.0;
const maxZoomLevel = 10.0;
const minZoomLevel = 0.1;
let smoothedLatitude = 0;
let smoothedLongitude = 0;
let smoothedZoom = 1.0;
let lastIndexPos = null;
let lastHandDetectedTime = Date.now();

// Hover state tracking
let isHoveringCountry = false;
let lastHoveredEntity = null;

// Safety bounds for coordinates
const SAFE_LAT_MIN = -85;
const SAFE_LAT_MAX = 85;
const SAFE_LON_MIN = -180;
const SAFE_LON_MAX = 180;
const SAFE_HEIGHT_MIN = 1000000; // 1M meters minimum
const SAFE_HEIGHT_MAX = 50000000; // 50M meters maximum

// Function to validate and clamp values
function safeValue(value, min, max, defaultValue = 0) {
    if (typeof value !== 'number' || isNaN(value) || !isFinite(value)) {
    return defaultValue;
    }
    return Math.max(min, Math.min(max, value));
}

// Exponential smoothing variables
const landmarkSmoothing = {
    alpha: 0.3,
    leftHand: null,
    rightHand: null
};

// Smoothing for pinch distance and position
let smoothedPinchDistance = 0;
let smoothedIndexPosition = null;
let smoothedRightHandPosition = null;
let lastRightHandY = null;
const pinchSmoothing = 0.4;
const positionSmoothing = 0.25;

const maxFontSize = 35;
const minFontSize = 1;
const minHeight = 200000;
const maxHeight = 20000000;

// Layer switching functionality
const layerProviders = {
    terrain: () => new Cesium.UrlTemplateImageryProvider({
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    maximumLevel: 18
    }),
    dark: () => new Cesium.UrlTemplateImageryProvider({
    url: 'https://cartodb-basemaps-{s}.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png',
    subdomains: 'abcd',
    maximumLevel: 18
    }),
    streets: () => new Cesium.OpenStreetMapImageryProvider({
    url: 'https://tile.openstreetmap.org/'
    })
};

function switchLayer(layerType) {
    try {
    // Remove existing imagery layers
    viewer.imageryLayers.removeAll();
    
    // Add new layer
    const provider = layerProviders[layerType]();
    viewer.imageryLayers.addImageryProvider(provider);
    
    // Update active state in UI
    document.querySelectorAll('.layer-option').forEach(option => {
        option.classList.remove('active');
    });
    document.querySelector(`input[value="${layerType}"]`).parentElement.classList.add('active');
    
    console.log(`Switched to ${layerType} layer`);
    } catch (error) {
    console.error(`Error switching to ${layerType} layer:`, error);
    // Fallback to terrain layer
    if (layerType !== 'terrain') {
        switchLayer('terrain');
    }
    }
}

function smoothLandmarks(landmarks, previousSmoothed, alpha) {
    if (!Array.isArray(landmarks) || landmarks.length < 21) {
    return previousSmoothed || [];
    }

    if (!previousSmoothed) {
    return landmarks.map(lm => lm ? { ...lm } : { x: 0, y: 0, z: 0 });
    }

    return landmarks.map((lm, i) => {
    if (!lm || !previousSmoothed[i]) {
        return lm ? { ...lm } : { x: 0, y: 0, z: 0 };
    }
    
    const x = typeof lm.x === 'number' ? lm.x : (previousSmoothed[i].x || 0);
    const y = typeof lm.y === 'number' ? lm.y : (previousSmoothed[i].y || 0);
    const z = typeof lm.z === 'number' ? lm.z : (previousSmoothed[i].z || 0);
    
    return {
        x: previousSmoothed[i].x * (1 - alpha) + x * alpha,
        y: previousSmoothed[i].y * (1 - alpha) + y * alpha,
        z: previousSmoothed[i].z * (1 - alpha) + z * alpha
    };
    });
}

function getDynamicFontSize() {
    const height = viewer.camera.positionCartographic.height;
    const t = Math.min(1, Math.max(0, (height - minHeight) / (maxHeight - minHeight)));
    const fontSize = minFontSize + (1 - t) * (maxFontSize - minFontSize);
    return `${fontSize.toFixed(1)}px sans-serif`;
}

function updateCanvasSize() {
    canvasElement.width = window.innerWidth;
    canvasElement.height = window.innerHeight;
}

async function initWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
    videoElement.srcObject = stream;
    return new Promise((resolve) => {
    videoElement.onloadedmetadata = () => {
        updateCanvasSize();
        resolve();
    };
    });
}

function initCesium() {
    Cesium.Ion.defaultAccessToken = cesiumToken;
    viewer = new Cesium.Viewer('cesiumContainer', {
    baseLayerPicker: false, fullscreenButton: false, geocoder: false,
    homeButton: false, infoBox: false, sceneModePicker: false,
    selectionIndicator: false, timeline: false, navigationHelpButton: false,
    animation: false, scene3DOnly: true, skyBox: false, skyAtmosphere: false,
    shouldAnimate: true,
    contextOptions: { webgl: { alpha: true } }
    });

    viewer.scene.backgroundColor = Cesium.Color.TRANSPARENT;
    viewer.scene.globe.enableLighting = false;
    viewer.scene.globe.baseColor = Cesium.Color.TRANSPARENT;

    camera = viewer.scene.camera;
    
    // Set initial safe position
    const initialHeight = 15000000;
    camera.setView({
    destination: Cesium.Cartesian3.fromDegrees(0, 0, initialHeight),
    orientation: { heading: 0, pitch: Cesium.Math.toRadians(-90), roll: 0 }
    });

    viewer.scene.screenSpaceCameraController.enableRotate = true;
    viewer.scene.screenSpaceCameraController.enableZoom = true;
    viewer.scene.screenSpaceCameraController.enableTranslate = false;
    viewer.scene.screenSpaceCameraController.enableTilt = false;

    infoBox = document.getElementById('infoBox');

    // Set up layer switching
    document.querySelectorAll('input[name="layer"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        switchLayer(e.target.value);
    });
    });

    // Load initial layer (terrain as default)
    switchLayer('terrain');

    // Load country boundaries with error handling
    Cesium.GeoJsonDataSource.load('https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson', {
    stroke: Cesium.Color.YELLOW,
    fill: Cesium.Color.TRANSPARENT, // Start with a transparent fill
    strokeWidth: 1
    }).then(function(dataSource) {
    viewer.dataSources.add(dataSource);

    const entities = dataSource.entities.values;
    for (let i = 0; i < entities.length; i++) {
        const entity = entities[i];
        const name = entity.name || entity.properties?.NAME?.getValue() || 'Unknown';

        // Skip entities that are just labels
        if (!entity.polygon) continue;

        // Store original styling for hover effects
        entity._originalStroke = Cesium.Color.YELLOW;
        entity._originalStrokeWidth = 1;
        entity._originalFill = Cesium.Color.TRANSPARENT;

        // Calculate the center for the label position
        const positions = entity.polygon.hierarchy.getValue(Cesium.JulianDate.now()).positions;
        let latSum = 0, lonSum = 0, count = 0;
        for (const pos of positions) {
            const cartographic = Cesium.Cartographic.fromCartesian(pos);
            const lat = Cesium.Math.toDegrees(cartographic.latitude);
            const lon = Cesium.Math.toDegrees(cartographic.longitude);
            if (isFinite(lat) && isFinite(lon)) {
                latSum += lat;
                lonSum += lon;
                count++;
            }
        }

        if (count > 0) {
            const lat = safeValue(latSum / count, SAFE_LAT_MIN, SAFE_LAT_MAX);
            const lon = safeValue(lonSum / count, SAFE_LON_MIN, SAFE_LON_MAX);

            // Add the label as a separate entity, but link it to the polygon entity
            const labelEntity = viewer.entities.add({
                position: Cesium.Cartesian3.fromDegrees(lon, lat),
                label: {
                    text: name,
                    font: getDynamicFontSize(),
                    fillColor: Cesium.Color.WHITE,
                    heightReference: Cesium.HeightReference.NONE,
                    outlineWidth: 0,
                    style: Cesium.LabelStyle.FILL
                },
                // Link back to the polygon entity
                _parentPolygon: entity 
            });

            // Also link from the polygon entity to the label
            entity._labelEntity = labelEntity;
        }
    }
    }).catch(error => {
    console.warn('Could not load country boundaries:', error);
    });

    // Update label fonts dynamically on zoom with error handling
    viewer.scene.postRender.addEventListener(() => {
    try {
        const font = getDynamicFontSize();
        viewer.entities.values.forEach(entity => {
        if (entity.label) {
            entity.label.font = font;
        }
        });
    } catch (error) {
        console.warn('Error updating label fonts:', error);
    }
    });

    // Add error handling for Cesium rendering errors
    viewer.scene.renderError.addEventListener(function(scene, error) {
    console.error('Cesium rendering error:', error);
    
    // Try to recover by resetting to safe position
    try {
        smoothedLatitude = 0;
        smoothedLongitude = 0;
        smoothedZoom = 1.0;
        currentZoomLevel = 1.0;
        targetZoomLevel = 1.0;
        
        camera.setView({
        destination: Cesium.Cartesian3.fromDegrees(0, 0, 15000000),
        orientation: { heading: 0, pitch: Cesium.Math.toRadians(-90), roll: 0 }
        });
    } catch (recoveryError) {
        console.error('Failed to recover from rendering error:', recoveryError);
    }
    });
}

function calculateDistance(a, b) {
    if (!a || !b || typeof a.x !== 'number' || typeof a.y !== 'number' || 
        typeof b.x !== 'number' || typeof b.y !== 'number') {
    return 0;
    }
    const dx = a.x - b.x, dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
}

function drawLandmarks(landmarks, isLeft, isPinching = false, isHovering = false) {
    if (!Array.isArray(landmarks) || landmarks.length < 21) return;
    try {
    const importantLandmarks = [4, 8];
    for (const i of importantLandmarks) {
        const lm = landmarks[i];
        if (!lm) continue;
        
        let radius = 6; // Default radius
        if (isPinching) {
            radius = 10;
        } else if (!isLeft && i === 8 && isHovering) {
            radius = 15; // Larger radius when hovering over country
        }
        
        canvasCtx.beginPath();
        canvasCtx.arc(lm.x * canvasElement.width, lm.y * canvasElement.height, radius, 0, 2 * Math.PI);
        
        if (isPinching) {
        canvasCtx.fillStyle = '#FF69B4';
        canvasCtx.fill();
        } else if (!isLeft && i === 8 && isHovering) {
        // Right hand index finger (landmark 8) turns pink and larger when hovering
        canvasCtx.fillStyle = '#FF69B4';
        canvasCtx.fill();
        canvasCtx.strokeStyle = '#000000';
        canvasCtx.lineWidth = 3; // Thicker stroke for larger circle
        canvasCtx.stroke();
        } else {
        canvasCtx.fillStyle = '#FFFFFF';
        canvasCtx.fill();
        canvasCtx.strokeStyle = '#000000';
        canvasCtx.lineWidth = 1;
        canvasCtx.stroke();
        }
    }

    const wrist = landmarks[0];
    if (wrist) {
        const labelY = wrist.y * canvasElement.height + 40;
        const labelX = wrist.x * canvasElement.width;
        
        canvasCtx.save();
        canvasCtx.translate(labelX, labelY);
        canvasCtx.scale(-1, 1);
        
        canvasCtx.font = '32px monospace';
        canvasCtx.fillStyle = '#FFFFFF';
        canvasCtx.strokeStyle = '#000000';
        canvasCtx.lineWidth = 3;
        canvasCtx.textAlign = 'center';
        
        const labelText = isLeft ? 'ROTATE ↻' : 'ZOOM ⭥';
        
        canvasCtx.strokeText(labelText, 0, 0);
        canvasCtx.fillText(labelText, 0, 0);
        
        canvasCtx.restore();
    }
    
    } catch (err) {
    console.warn('Error in drawLandmarks:', err);
    }
}

async function initMediaPipeHands() {
    const hands = new Hands({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` });
    hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 0,
    minDetectionConfidence: 0.4,
    minTrackingConfidence: 0.4,
    });
    await hands.initialize();
    return hands;
}

// Safe camera update function
function updateCameraPosition(lon, lat, zoom) {
    try {
    // Validate all inputs
    const safeLon = (typeof lon === 'number' && isFinite(lon)) ? lon : 0;
    const safeLat = safeValue(lat, SAFE_LAT_MIN, SAFE_LAT_MAX, 0);
    const safeZoom = safeValue(zoom, minZoomLevel, maxZoomLevel, 1.0);
    const safeHeight = safeValue(15000000 * safeZoom, SAFE_HEIGHT_MIN, SAFE_HEIGHT_MAX, 15000000);

    // Create destination with validated values
    const destination = Cesium.Cartesian3.fromDegrees(safeLon, safeLat, safeHeight);
    
    // Validate the destination
    if (!destination || !isFinite(destination.x) || !isFinite(destination.y) || !isFinite(destination.z)) {
        console.warn('Invalid destination calculated, skipping camera update');
        return false;
    }

    camera.setView({
        destination: destination,
        orientation: {
        heading: 0,
        pitch: Cesium.Math.toRadians(-90),
        roll: 0
        }
    });
    
    return true;
    } catch (error) {
    console.error('Error updating camera position:', error);
    return false;
    }
}

async function handleCountryHover(landmarks) {
    const infoBox = document.getElementById('infoBox');

    // If no landmarks are provided, clear any existing hover effect and hide the info box.
    if (!landmarks || landmarks.length < 9) {
        if (lastHoveredEntity && lastHoveredEntity.polygon) {
            // Reset to original styling
            lastHoveredEntity.polygon.material = lastHoveredEntity._originalFill;
            lastHoveredEntity.polygon.outline = true;
            lastHoveredEntity.polygon.outlineColor = lastHoveredEntity._originalStroke;
            lastHoveredEntity.polygon.outlineWidth = lastHoveredEntity._originalStrokeWidth;
        }
        lastHoveredEntity = null;
        infoBox.style.display = 'none';
        isHoveringCountry = false;
        return;
    }

    const indexTip = landmarks[8];
    const screenPosition = new Cesium.Cartesian2((1 - indexTip.x) * canvasElement.width, indexTip.y * canvasElement.height);
    const pickedObject = viewer.scene.pick(screenPosition);

    let currentEntity = null;
    if (pickedObject && pickedObject.id) {
        // If we picked a label, get its parent polygon entity
        if (pickedObject.id._parentPolygon) {
            currentEntity = pickedObject.id._parentPolygon;
        } else {
            currentEntity = pickedObject.id;
        }
    }

    if (currentEntity && currentEntity.polygon) {
        isHoveringCountry = true;
        
        // If the finger is over a new country, update the info.
        if (currentEntity !== lastHoveredEntity) {
            // Reset the style of the previously hovered country.
            if (lastHoveredEntity && lastHoveredEntity.polygon) {
                lastHoveredEntity.polygon.material = lastHoveredEntity._originalFill;
                lastHoveredEntity.polygon.outline = true;
                lastHoveredEntity.polygon.outlineColor = lastHoveredEntity._originalStroke;
                lastHoveredEntity.polygon.outlineWidth = lastHoveredEntity._originalStrokeWidth;
            }

            lastHoveredEntity = currentEntity;

            // Apply pink hover styling
            currentEntity.polygon.material = Cesium.Color.HOTPINK.withAlpha(0.3);
            currentEntity.polygon.outline = true;
            currentEntity.polygon.outlineColor = Cesium.Color.HOTPINK;
            currentEntity.polygon.outlineWidth = 3; // Thicker border
            
            const countryName = currentEntity.name || currentEntity.properties?.NAME?.getValue();

            if (countryName) {
                try {
                    const response = await fetch(`https://restcountries.com/v3.1/name/${countryName}?fields=name,capital,population,currencies,region,flags`);
                    const data = await response.json();
                    if (data.length > 0) {
                        const country = data[0];
                        const currency = Object.values(country.currencies)[0];
                        infoBox.innerHTML = `<img src="${country.flags.svg}" alt="Flag of ${country.name.common}" style="width:100px;border: 2px solid #000;"><br><strong>${country.name.common}</strong><br>Capital: ${country.capital[0]}<br>Population: ${country.population.toLocaleString()}<br>Currency: ${currency.name} (${currency.symbol})<br>Region: ${country.region}`;
                        infoBox.style.display = 'block';
                    }
                } catch (error) {
                    console.error('Error fetching country data:', error);
                    infoBox.innerHTML = `<strong>${countryName}</strong><br>Could not fetch data.`;
                    infoBox.style.display = 'block';
                }
            }
        }

    } else {
        isHoveringCountry = false;
        
        // If the finger is not over any country, reset the hover effect.
        if (lastHoveredEntity && lastHoveredEntity.polygon) {
            lastHoveredEntity.polygon.material = lastHoveredEntity._originalFill;
            lastHoveredEntity.polygon.outline = true;
            lastHoveredEntity.polygon.outlineColor = lastHoveredEntity._originalStroke;
            lastHoveredEntity.polygon.outlineWidth = lastHoveredEntity._originalStrokeWidth;
        }
        lastHoveredEntity = null;
        infoBox.style.display = 'none';
    }
}

function onResults(results) {
    try {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    updateCanvasSize();

    const now = Date.now();
    let rightDetected = false;
    let leftDetected = false;

    if (!results || !results.multiHandLandmarks || !Array.isArray(results.multiHandLandmarks)) {
        return;
    }

    if (results.multiHandLandmarks.length > 0) {
        lastHandDetectedTime = now;

        for (let i = 0; i < results.multiHandLandmarks.length; i++) {
            try {
            const rawLandmarks = results.multiHandLandmarks[i];
            
            if (!Array.isArray(rawLandmarks) || rawLandmarks.length < 21) {
                console.warn(`Invalid landmarks for hand ${i}:`, rawLandmarks);
                continue;
            }

            if (!results.multiHandedness || !results.multiHandedness[i] || !results.multiHandedness[i].label) {
                console.warn(`Invalid handedness data for hand ${i}`);
                continue;
            }

            const isLeft = results.multiHandedness[i].label === 'Left';

            let smoothedLandmarks;
            if (isLeft) {
                landmarkSmoothing.leftHand = smoothLandmarks(
                    rawLandmarks, 
                    landmarkSmoothing.leftHand, 
                    landmarkSmoothing.alpha
                );
                smoothedLandmarks = landmarkSmoothing.leftHand;
            } else {
                landmarkSmoothing.rightHand = smoothLandmarks(
                    rawLandmarks, 
                    landmarkSmoothing.rightHand, 
                    landmarkSmoothing.alpha
                );
                smoothedLandmarks = landmarkSmoothing.rightHand;
            }

            if (!smoothedLandmarks || smoothedLandmarks.length < 21) {
                console.warn(`Invalid smoothed landmarks for hand ${i}`);
                continue;
            }

            let isPinching = false;

            if (!isLeft) {
                const thumb = smoothedLandmarks[4];
                const index = smoothedLandmarks[8];
                
                if (!thumb || !index) {
                    console.warn('Missing thumb or index finger landmark for right hand');
                    continue;
                }

                const rawPinch = calculateDistance(thumb, index);
                smoothedPinchDistance = smoothedPinchDistance * (1 - pinchSmoothing) + rawPinch * pinchSmoothing;
                
                if (smoothedPinchDistance < 0.08) {
                    isPinching = true;
                    const indexTip = smoothedLandmarks[8];
                    
                    if (!indexTip || typeof indexTip.x !== 'number' || typeof indexTip.y !== 'number') {
                        console.warn('Invalid index tip position for zoom');
                        continue;
                    }
                    
                    if (!smoothedRightHandPosition) {
                        smoothedRightHandPosition = { x: indexTip.x, y: indexTip.y };
                    } else {
                        smoothedRightHandPosition.x = smoothedRightHandPosition.x * (1 - positionSmoothing) + indexTip.x * positionSmoothing;
                        smoothedRightHandPosition.y = smoothedRightHandPosition.y * (1 - positionSmoothing) + indexTip.y * positionSmoothing;
                    }
                    
                    if (lastRightHandY !== null) {
                        const deltaY = smoothedRightHandPosition.y - lastRightHandY;
                        // Reduced zoom speed for more stability
                        const zoomSpeed = 5.0;
                        const newTargetZoom = targetZoomLevel + deltaY * zoomSpeed;
                        targetZoomLevel = safeValue(newTargetZoom, minZoomLevel, maxZoomLevel, targetZoomLevel);
                    }
                    lastRightHandY = smoothedRightHandPosition.y;
                    rightDetected = true;
                } else {
                    lastRightHandY = null;
                    smoothedRightHandPosition = null;
                }

                // Always call handleCountryHover when the right hand is visible
                handleCountryHover(smoothedLandmarks);
            } else {
                const indexTip = smoothedLandmarks[8];
                const thumbTip = smoothedLandmarks[4];
                
                if (!indexTip || !thumbTip) {
                    console.warn('Missing index or thumb landmark for left hand');
                    continue;
                }

                const pinchDistance = calculateDistance(indexTip, thumbTip);

                if (pinchDistance < 0.08) {
                    isPinching = true;
                    if (typeof indexTip.x !== 'number' || typeof indexTip.y !== 'number') {
                        console.warn('Invalid index tip coordinates for navigation');
                        continue;
                    }

                    if (!smoothedIndexPosition) {
                        smoothedIndexPosition = { x: indexTip.x, y: indexTip.y };
                    } else {
                        smoothedIndexPosition.x = smoothedIndexPosition.x * (1 - positionSmoothing) + indexTip.x * positionSmoothing;
                        smoothedIndexPosition.y = smoothedIndexPosition.y * (1 - positionSmoothing) + indexTip.y * positionSmoothing;
                    }

                    if (lastIndexPos) {
                        const deltaX = smoothedIndexPosition.x - lastIndexPos.x;
                        const deltaY = smoothedIndexPosition.y - lastIndexPos.y;
                        
                        // Add bounds checking for deltas
                        if (typeof deltaX === 'number' && typeof deltaY === 'number' && 
                            Math.abs(deltaX) < 0.5 && Math.abs(deltaY) < 0.5 &&
                            isFinite(deltaX) && isFinite(deltaY)) {
                            
                            // Reduced rotation speed for more stability
                            const baseRotationSpeed = 240;
                            const zoomFactor = Math.max(0.1, Math.min(2.0, smoothedZoom));
                            const rotationSpeed = baseRotationSpeed * zoomFactor;
                            
                            const newLon = smoothedLongitude + deltaX * rotationSpeed;
                            const newLat = smoothedLatitude + deltaY * (rotationSpeed * 0.5);
                            
                            smoothedLongitude = newLon;
                            smoothedLatitude = safeValue(newLat, SAFE_LAT_MIN, SAFE_LAT_MAX, smoothedLatitude);
                        }
                    }
                    lastIndexPos = { x: smoothedIndexPosition.x, y: smoothedIndexPosition.y };
                    leftDetected = true;
                } else {
                    lastIndexPos = null;
                    smoothedIndexPosition = null;
                }
            }

            // Draw landmarks with hover state for right hand
            drawLandmarks(smoothedLandmarks, isLeft, isPinching, !isLeft && isHoveringCountry);
            } catch (handError) {
            console.warn(`Error processing hand ${i}:`, handError);
            continue;
            }
        }
    } else {
        landmarkSmoothing.leftHand = null;
        landmarkSmoothing.rightHand = null;
        smoothedIndexPosition = null;
        smoothedRightHandPosition = null;
        lastRightHandY = null;
        handleCountryHover(null); // Clear hover when no hands are detected
    }

    // Update zoom with validation
    if (now - lastHandDetectedTime < 500) {
        if (typeof targetZoomLevel === 'number' && isFinite(targetZoomLevel)) {
        const newZoom = currentZoomLevel + (targetZoomLevel - currentZoomLevel) * 0.1;
        currentZoomLevel = safeValue(newZoom, minZoomLevel, maxZoomLevel, currentZoomLevel);
        smoothedZoom = currentZoomLevel;
        }
    }

    // Update camera position with validation
    if (typeof smoothedLongitude === 'number' && typeof smoothedLatitude === 'number' && 
        typeof smoothedZoom === 'number' && isFinite(smoothedLongitude) && 
        isFinite(smoothedLatitude) && isFinite(smoothedZoom)) {
        
        updateCameraPosition(smoothedLongitude, smoothedLatitude, smoothedZoom);
    }
    
    } catch (error) {
    console.error('Error in onResults:', error);
    
    // Reset to safe state on error
    landmarkSmoothing.leftHand = null;
    landmarkSmoothing.rightHand = null;
    smoothedIndexPosition = null;
    smoothedRightHandPosition = null;
    lastIndexPos = null;
    lastRightHandY = null;
    
    // Reset position values to safe defaults
    smoothedLatitude = safeValue(smoothedLatitude, SAFE_LAT_MIN, SAFE_LAT_MAX, 0);
    smoothedLongitude = (typeof smoothedLongitude === 'number' && isFinite(smoothedLongitude)) ? smoothedLongitude : 0;
    smoothedZoom = safeValue(smoothedZoom, minZoomLevel, maxZoomLevel, 1.0);
    currentZoomLevel = smoothedZoom;
    targetZoomLevel = smoothedZoom;
    isHoveringCountry = false;
    }
}

async function startApp() {
    try {
    await initWebcam();
    initCesium();
    const hands = await initMediaPipeHands();
    hands.onResults(onResults);

    const cameraUtils = new Camera(videoElement, {
        onFrame: async () => {
        try {
            await hands.send({ image: videoElement });
        } catch (error) {
            console.warn('Error sending frame to MediaPipe:', error);
        }
        },
        width: 1280, height: 720
    });
    cameraUtils.start();
    } catch (error) {
    console.error('Error starting application:', error);
    alert('Failed to start the application. Please check your camera permissions and refresh the page.');
    }
}

// Add window error handler
window.addEventListener('error', function(event) {
    console.error('Global error:', event.error);
    
    // If it's a Cesium-related error, try to reset the camera
    if (event.error && event.error.message && event.error.message.includes('Array')) {
    try {
        smoothedLatitude = 0;
        smoothedLongitude = 0;
        smoothedZoom = 1.0;
        currentZoomLevel = 1.0;
        targetZoomLevel = 1.0;
        isHoveringCountry = false;
        
        if (camera) {
        updateCameraPosition(0, 0, 1.0);
        }
    } catch (resetError) {
        console.error('Failed to reset after global error:', resetError);
    }
    }
});

startApp();