/**
 * loading_cursor.js
 *
 * Shows a "wait" cursor globally while any Dash callback is in flight.
 * Dash uses fetch() for all /_dash-update-component calls, so we patch
 * window.fetch to track pending requests and update document.body.style.cursor.
 */
(function () {
    var pendingCallbacks = 0;

    var _originalFetch = window.fetch;

    window.fetch = function (url) {
        var isDashCallback =
            typeof url === "string" &&
            url.indexOf("/_dash-update-component") !== -1;

        if (isDashCallback) {
            pendingCallbacks++;
            document.body.style.cursor = "wait";
        }

        var result = _originalFetch.apply(this, arguments);

        if (isDashCallback) {
            result.finally(function () {
                pendingCallbacks = Math.max(0, pendingCallbacks - 1);
                if (pendingCallbacks === 0) {
                    document.body.style.cursor = "";
                }
            });
        }

        return result;
    };
})();
