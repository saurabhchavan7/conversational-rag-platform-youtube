/**
 * Content Script - Runs on YouTube Pages
 * Extracts video ID from current YouTube page
 */

/**
 * Get current YouTube video ID from URL
 * @returns {string|null} Video ID or null if not on video page
 */
function getCurrentVideoId() {
    // Check if we're on a YouTube video page
    const url = window.location.href;
    
    if (!url.includes('youtube.com/watch')) {
        return null;
    }
    
    // Extract video ID from URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const videoId = urlParams.get('v');
    
    return videoId;
}

/**
 * Get video title from page
 * @returns {string} Video title
 */
function getVideoTitle() {
    const titleElement = document.querySelector('h1.ytd-video-primary-info-renderer') ||
                         document.querySelector('h1.title') ||
                         document.querySelector('ytd-watch-metadata #title h1');
    
    return titleElement ? titleElement.textContent.trim() : 'Unknown Video';
}

/**
 * Listen for messages from popup
 */
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'getVideoInfo') {
        const videoId = getCurrentVideoId();
        const videoTitle = getVideoTitle();
        
        sendResponse({
            videoId: videoId,
            videoTitle: videoTitle,
            isYouTubePage: videoId !== null
        });
    }
    
    return true;  // Keep message channel open for async response
});

// Log when content script loads
console.log('YouTube RAG Chat: Content script loaded');